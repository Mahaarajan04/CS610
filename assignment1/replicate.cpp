#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "headers.h"
using namespace std;
int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <N>\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];
    int N = std::atoi(argv[3]);

    if (N <= 0) {
        std::cerr << "Error: N must be a positive integer.\n";
        return 1;
    }

    std::ifstream in(input_path);
    if (!in) {
        std::cerr << "Error: Cannot open input file.\n";
        return 1;
    }

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Error: Cannot open output file.\n";
        return 1;
    }

    std::string line;
    int count = 0;
    while (std::getline(in, line)) {
        for (int i = 0; i < N; ++i) {
            out << line + "  "+ to_string(count++) << '\n';
        }
    }

    std::cout << "Successfully replicated each line " << N << " times.\n";
    return 0;
}
