#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <csignal>
#include <thread>
//hehehe
using std::cerr;
using std::cout;
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define N (1e7) //og was 1e7
#define NUM_THREADS (8)

std::atomic<bool> timeout_flag{false};


// Shared variables
uint64_t var1 = 0, var2 = (N * NUM_THREADS + 1);
//Inline assembly functions for atomic operations
inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
__asm__ __volatile__("pause");
#else
__asm__ __volatile__("" ::: "memory");
#endif
}

struct alignas(64) PaddedInt { int val; };
struct alignas(64) PaddedBool { bool val; };

inline void memory_barrier() {
    asm volatile ("mfence" ::: "memory");
}

inline int atomic_lock_test_and_set(volatile int* ptr, int val = 1) {
    int old;
    asm volatile (
        "xchg %0, %1"
        : "=r"(old), "+m"(*ptr)
        : "0"(val)
        : "memory"
    );
    return old;
}

inline int atomic_fetch_and_add(volatile int* ptr, int val = 1) {
    int old;
    asm volatile (
        "lock xadd %0, %1"
        : "=r"(old), "+m"(*ptr)
        : "0"(val)
        : "memory"
    );
    return old;
}

inline int atomic_compare_and_swap(volatile int* ptr, int expected, int desired) {
    int old;
    asm volatile (
        "lock cmpxchg %2, %1"
        : "=a"(old), "+m"(*ptr)
        : "r"(desired), "0"(expected)
        : "memory"
    );
    return old;
}

class LockBase {
public:
    virtual void acquire(uint16_t tid) = 0;
    virtual void release(uint16_t tid) = 0;
};

typedef struct thr_args {
    uint16_t m_id;
    LockBase* m_lock;
} ThreadArgs;

class PthreadMutex : public LockBase {
public:
    void acquire(uint16_t tid) override { pthread_mutex_lock(&lock); }
    void release(uint16_t tid) override { pthread_mutex_unlock(&lock); }
private:
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
};

class FilterLock : public LockBase {
private:
    PaddedInt level[NUM_THREADS];
    PaddedInt victim[NUM_THREADS];
public:
    void acquire(uint16_t tid) override {
        for (int i = 1; i < NUM_THREADS; i++) {
            level[tid].val = i;
            memory_barrier();
            victim[i].val = tid;
            memory_barrier();
            for (int k = 0; k < NUM_THREADS; k++) {
                if (k == tid) continue;
                while (level[k].val >= i && victim[i].val == tid) cpu_relax();
            }
        }
    }
    void release(uint16_t tid) override {
        memory_barrier();
        level[tid].val = 0;
    }
};

class BakeryLock : public LockBase {
private:
    PaddedBool choosing[NUM_THREADS];
    PaddedInt ticket[NUM_THREADS];
public:
    void acquire(uint16_t tid) override {
        choosing[tid].val = true;
        memory_barrier();
        int max_ticket = 0;
        for (int k = 0; k < NUM_THREADS; k++) {
            if (ticket[k].val > max_ticket) max_ticket = ticket[k].val;
        }
        ticket[tid].val = max_ticket + 1;
        memory_barrier();
        choosing[tid].val = false;
        memory_barrier();
        for (int k = 0; k < NUM_THREADS; k++) {
            if (k == tid) continue;
            while (choosing[k].val) cpu_relax();
            while (ticket[k].val != 0 && (ticket[k].val < ticket[tid].val || (ticket[k].val == ticket[tid].val && k < tid))) cpu_relax();
        }
    }
    void release(uint16_t tid) override {
        memory_barrier();
        ticket[tid].val = 0;
    }
};

class SpinLock : public LockBase {
private:
    alignas(64) volatile bool flag = false;
public:
    void acquire(uint16_t tid) override {
        while (atomic_lock_test_and_set((volatile int*)&flag)) cpu_relax();
        memory_barrier();
    }
    void release(uint16_t tid) override {
        memory_barrier();
        flag = false;
    }
};

class TicketLock : public LockBase {
private:
    alignas(64) volatile int next_ticket = 0;
    alignas(64) volatile int serving_now = 0;
public:
    void acquire(uint16_t tid) override {
        int my_ticket = atomic_fetch_and_add(&next_ticket, 1);
        while (serving_now != my_ticket) cpu_relax();
        memory_barrier();
    }
    void release(uint16_t tid) override {
        memory_barrier();
        atomic_fetch_and_add(&serving_now, 1);
    }
};

class ArrayQLock : public LockBase {
private:
    static constexpr int SIZE = NUM_THREADS;
    struct alignas(64) FlagSlot { volatile bool flag; };
    FlagSlot* flags;
    int* mySlot;
    alignas(64) volatile int tail = 0;
public:
    void acquire(uint16_t tid) override {
        int slot = atomic_fetch_and_add(&tail, 1) % SIZE;
        mySlot[tid] = slot;
        while (!flags[slot].flag) cpu_relax();
        flags[slot].flag = false;
        memory_barrier();
    }
    void release(uint16_t tid) override {
        int slot = mySlot[tid];
        int next = (slot + 1) % SIZE;
        memory_barrier();
        flags[next].flag = true;
    }
    ArrayQLock() {
        flags = new FlagSlot[SIZE];
        mySlot = new int[SIZE];
        for (int i = 0; i < SIZE; ++i) flags[i].flag = false;
        flags[0].flag = true;
    }
    ~ArrayQLock() {
        delete[] flags;
        delete[] mySlot;
    }
};


/** Estimate the time taken */
std::atomic_uint64_t sync_time = 0;

inline void critical_section() {
  var1++;
  var2--;
}

/** Sync threads at the start to maximize contention */
pthread_barrier_t g_barrier;

void* thrBody(void* arguments) {
  ThreadArgs* tmp = static_cast<ThreadArgs*>(arguments);
  if (false) {
    cout << "Thread id: " << tmp->m_id << " starting\n";
  }

  // Wait for all other producer threads to launch before proceeding.
  pthread_barrier_wait(&g_barrier);

  HRTimer start = HR::now();
  for (int i = 0; i < N; i++) {
    tmp->m_lock->acquire(tmp->m_id);
    critical_section();
    tmp->m_lock->release(tmp->m_id);
  }
  HRTimer end = HR::now();
  auto duration = duration_cast<milliseconds>(end - start).count();

  // A barrier is not required here
  sync_time.fetch_add(duration);
  pthread_exit(NULL);
}

// Timeout handler
void timeout_watchdog() {
  std::this_thread::sleep_for(std::chrono::seconds(120));
  timeout_flag = true;
  std::cerr << "Execution timed out after 2 minutes.\n";
  std::exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: ./executable <lock_type>\n";
    std::cerr << "Valid options: pthread, filter, bakery, spin, ticket, arrayq\n";
    return EXIT_FAILURE;
  }

  std::string lock_type(argv[1]);

  std::thread watchdog(timeout_watchdog); // Start the timeout thread

  int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
  if (error != 0) {
    std::cerr << "Error in barrier init.\n";
    return EXIT_FAILURE;
  }

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t tid[NUM_THREADS];
  ThreadArgs args[NUM_THREADS] = {{0}};

  LockBase* lock_obj = nullptr;

  if (lock_type == "pthread") lock_obj = new PthreadMutex();
  else if (lock_type == "filter") lock_obj = new FilterLock();
  else if (lock_type == "bakery") lock_obj = new BakeryLock();
  else if (lock_type == "spin") lock_obj = new SpinLock();
  else if (lock_type == "ticket") lock_obj = new TicketLock();
  else if (lock_type == "arrayq") lock_obj = new ArrayQLock();
  else {
    std::cerr << "Invalid lock type\n";
    return EXIT_FAILURE;
  }

  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  for (int i = 0; i < NUM_THREADS; ++i) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;
    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      std::cerr << "\nThread cannot be created : " << strerror(error) << "\n";
      return EXIT_FAILURE;
    }
  }

  void* status;
  for (int i = 0; i < NUM_THREADS; ++i) {
    error = pthread_join(tid[i], &status);
    if (error) {
      std::cerr << "ERROR: return code from pthread_join() is " << error << "\n";
      return EXIT_FAILURE;
    }
  }

  watchdog.detach(); // Allow the timeout thread to exit naturally

  if (timeout_flag.load()) {
    std::cerr << "Terminating due to timeout\n";
    return EXIT_FAILURE;
  }

  std::cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  std::cout << lock_type << " lock: Time taken (us): " << sync_time << "\n";
  return 0;
  pthread_exit(NULL);
}
