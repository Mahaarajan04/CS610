#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <pthread.h>
using std::endl;

using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::milliseconds;

#define N (1e7) //og was 1e7
#define NUM_THREADS (8)

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

int main() {
  int error = pthread_barrier_init(&g_barrier, NULL, NUM_THREADS);
  if (error != 0) {
    cerr << "Error in barrier init.\n";
    exit(EXIT_FAILURE);
  }

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  pthread_t tid[NUM_THREADS];
  ThreadArgs args[NUM_THREADS] = {{0}};

  // Pthread mutex
  LockBase* lock_obj = new PthreadMutex();
  uint16_t i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      cerr << "\nThread cannot be created : " << strerror(error) << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  void* status;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      cerr << "ERROR: return code from pthread_join() is " << error << "\n";
      exit(EXIT_FAILURE);
    }
    i++;
  }

  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  cout << "Pthread mutex: Time taken (us): " << sync_time << "\n";

  // Filter lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new FilterLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Filter lock: Time taken (us): " << sync_time << "\n";
  // Bakery lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new BakeryLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Bakery lock: Time taken (us): " << sync_time << "\n";

  // Spin lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new SpinLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Spin lock: Time taken (us): " << sync_time << "\n";
  // Ticket lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new TicketLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Ticket lock: Time taken (us): " << sync_time << "\n";

  // Array Q lock
  var1 = 0;
  var2 = (N * NUM_THREADS + 1);
  sync_time.store(0);

  lock_obj = new ArrayQLock();
  i = 0;
  while (i < NUM_THREADS) {
    args[i].m_id = i;
    args[i].m_lock = lock_obj;

    error = pthread_create(&tid[i], &attr, thrBody, (void*)(args + i));
    if (error != 0) {
      printf("\nThread cannot be created : [%s]", strerror(error));
      exit(EXIT_FAILURE);
    }
    i++;
  }

  i = 0;
  while (i < NUM_THREADS) {
    error = pthread_join(tid[i], &status);
    if (error) {
      printf("ERROR: return code from pthread_join() is %d\n", error);
      exit(EXIT_FAILURE);
    }
    i++;
  }

  cout << "Var1: " << var1 << "\tVar2: " << var2 << "\n";
  assert(var1 == N * NUM_THREADS && var2 == 1);
  cout << "Array Q lock: Time taken (us): " << sync_time << "\n";

  pthread_barrier_destroy(&g_barrier);
  pthread_attr_destroy(&attr);

  pthread_exit(NULL);
  
}
