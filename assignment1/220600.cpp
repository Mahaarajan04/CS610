#include "220600.h"
using namespace std;

// Global shared variables
int num_lines_left_input = 0;
int num_lines_written_output = 0;
int total_lines = 0;
queue<string> shared_buffer;


// Synchronization primitives
mutex in_lock, buff_lock, buff_lock_prod_only, buff_lock_cons_only,out_lock;
condition_variable buffer_not_full, buffer_not_empty;

//boolean flags
bool producers_done = false;
bool thread_done = false;
bool input_file_ends_with_nl = true;

//newline fix
bool input_ends_with_newline(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    if (!fin) return true; // treat missing file as newline-terminated

    std::streampos size = fin.tellg();
    if (size == 0) return false;

    fin.seekg(-1, std::ios::end);
    char last;
    fin.get(last);
    return last == '\n';
}



void producer(int id, int num_threads, int Lmin, int Lmax, int M, std::ifstream& in) {
    while (true) {
        std::vector<string> lines;
        int L= rand() % (Lmax - Lmin + 1) + Lmin;

        // Critical section 1 : Read L lines from input under mutex
        {
            unique_lock<mutex> lk_in(in_lock);
            if (num_lines_left_input == 0){
                lk_in.unlock(); // Release lock before returning
                return;
            }
            for (int i = 0; i < L && num_lines_left_input > 0; i++) {
                string line;
                if (!getline(in, line)) break;
                lines.push_back(line); // Append thread ID for uniqueness
                num_lines_left_input--;
            }
            lk_in.unlock(); // Release lock after reading
        }


        // Critical section 2: Write to shared buffer under mutex
        int index = 0;
        {
            unique_lock<mutex> lk_prod(buff_lock_prod_only);
            while (index < lines.size()) {
                // Wait until buffer has space and singular access to shared_buffer
                unique_lock<mutex> lk(buff_lock);
                buffer_not_full.wait(lk, [&] {
                    return shared_buffer.size() < M;
                });

                while (index < lines.size() && shared_buffer.size() < M) {
                    shared_buffer.push(lines[index++]);
                }
                if(index!= lines.size()){
                    //cout<<"yesssss"<<endl;
                    thread_done = false; // If not all lines were added, set thread_done to false
                }
                else thread_done = true; // If all lines were added, set thread_done to true
                lk.unlock(); // Release lock after writing to buffer if full or writing done
                buffer_not_empty.notify_all();  // Wake up consumers
            }
            lk_prod.unlock(); // Release lock after writing all lines
        }
    }
}

void consumer(int id, int num_threads, int M, std::ofstream& out) {
    while (true) {
        vector<string> lines;

        // Critical section 1: Read from shared buffer under mutex
        unique_lock<mutex> lk_buff(buff_lock_cons_only);
        int i=0;
        while(!thread_done|| i==0){
            {
                //cout<<"Hiii"<<endl;
                unique_lock<mutex> lk(buff_lock);
                buffer_not_empty.wait(lk, [&] {
                    return !shared_buffer.empty() || (producers_done && shared_buffer.empty());
                });

                if (shared_buffer.empty() && producers_done) {
                    lk.unlock(); // Release lock before returning
                    lk_buff.unlock(); // Release lock before returning
                    buffer_not_empty.notify_all();
                    return;
                }

                while (!shared_buffer.empty()) {
                    lines.push_back(shared_buffer.front());
                    shared_buffer.pop();
                }
                i=1;
                lk.unlock(); // Release lock after reading from buffer
                if(!producers_done) buffer_not_full.notify_all();  // Wake up producers
                else buffer_not_empty.notify_all(); // Notify consumers if done reading for exits
            }
        }
        lk_buff.unlock(); // Release lock after reading all lines of the producer
        

        // Critical section 2: Write to output file under mutex
        {
            unique_lock<mutex> lk(out_lock);
            for (size_t i = 0; i < lines.size(); ++i) {
                if (i == lines.size() - 1 && !input_file_ends_with_nl && num_lines_written_output + 1 == total_lines) {
                    out << lines[i]; // no newline on last line if input didn't have it
                } else {
                    out << lines[i] << endl;
                }
                num_lines_written_output++;
            }
            lk.unlock(); // Release lock after writing
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        cout << "Usage: " << argv[0]
             << " <input_file> <num_threads> <Lmin> <Lmax> <buff_size> <output_file>" << endl;
        return -1;
    }

    // Parse command line arguments
    string input_file = argv[1];
    int num_threads = stoi(argv[2]);
    int Lmin = stoi(argv[3]);
    int Lmax = stoi(argv[4]);
    int M = stoi(argv[5]);
    string output_file = argv[6];

    // Validate arguments
    if (num_threads <= 0 || Lmin < 0 || Lmax < Lmin || M <= 0 || Lmax==0) {
        cout << "Invalid arguments provided." << endl;
        return -1;
    }
    input_file_ends_with_nl = input_ends_with_newline(input_file);
    std::ifstream in(input_file);
    std::ofstream out(output_file);
    string line;

    if (!in.is_open()) {
        perror("Error opening input file");
        return -1;
    }
    if (!out.is_open()) {
        perror("Error opening output file");
        return -1;
    }

    while (getline(in, line)) num_lines_left_input++;
    total_lines = num_lines_left_input;

    in.close();
    in.open(input_file);
    if (!in.is_open()) {
        perror("Error reopening input file");
        return -1;
    }

    // Launch threads
    vector<thread> producers, consumers;
    int num_consumers = max(1, num_threads / 2);

    for (int i = 0; i < num_threads; i++) producers.emplace_back(producer, i, num_threads, Lmin, Lmax, M, ref(in));

    for (int i = 0; i < num_consumers; i++) consumers.emplace_back(consumer, i, num_consumers, M, ref(out));

    for (auto& p : producers) p.join();

    // Signal that producers are done
    {
        unique_lock<mutex> lk(buff_lock);
        producers_done = true;
        lk.unlock();
    }
    buffer_not_empty.notify_all();  // Wake consumers waiting on buffer

    for (auto& c : consumers) c.join();

    //If the last character of inputfile is not a newline
    

    // close input and output files
    in.close();
    out.close();

    cout << "Total lines written: " << num_lines_written_output << endl;
    cout << "Total lines read: " << total_lines << endl;

    // Clean up
    producers.clear();
    consumers.clear();

    cout << "Processing complete." << endl;
    return 0;
}