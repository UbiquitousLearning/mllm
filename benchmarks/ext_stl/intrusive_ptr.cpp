
#include <memory>
#include <random>
#include <vector>
#include <thread>
#include <functional>
#include <benchmark/benchmark.h>

#include "mllm/utils/IntrusivePtr.hpp"

using namespace mllm;  // NOLINT

class TestObject : public RefCountedBase {
 public:
  explicit TestObject(int value = 0) : data(value) {}

  [[nodiscard]] int getData() const { return data; }

  void setData(int value) { data = value; }

  virtual void doWork() {
    // Simulate some work
    data = (data * 13 + 7) % 100;
  }

 private:
  int data;
};

// Class for std::shared_ptr testing
class TestObjectShared {
 public:
  explicit TestObjectShared(int value = 0) : data(value) {}
  [[nodiscard]] int getData() const { return data; }
  void setData(int value) { data = value; }

  virtual void doWork() {
    // Simulate some work
    data = (data * 13 + 7) % 100;
  }

 private:
  int data;
};

// Derived class for testing inheritance scenarios
class DerivedObject : public TestObject {
 public:
  DerivedObject(int value, double extra) : TestObject(value), extra_data(extra) {}

  void doWork() override {
    TestObject::doWork();
    extra_data = extra_data * 0.95 + 0.05;
  }

 private:
  double extra_data;
};

// Benchmark test: creation and destruction
static void BM_IntrusivePtr_CreateDestroy(benchmark::State& state) {
  const int count = state.range(0);
  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      IntrusivePtr<TestObject> ptr = make_intrusive<TestObject>(i);
      benchmark::DoNotOptimize(ptr);
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_IntrusivePtr_CreateDestroy)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SharedPtr_CreateDestroy(benchmark::State& state) {
  const int count = state.range(0);
  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      std::shared_ptr<TestObjectShared> ptr = std::make_shared<TestObjectShared>(i);
      benchmark::DoNotOptimize(ptr);
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_SharedPtr_CreateDestroy)->Arg(100)->Arg(1000)->Arg(10000);

// Benchmark test: copy operations
static void BM_IntrusivePtr_Copy(benchmark::State& state) {
  const int count = state.range(0);
  std::vector<IntrusivePtr<TestObject>> originals(count);
  for (int i = 0; i < count; ++i) { originals[i] = make_intrusive<TestObject>(i); }

  for (auto _ : state) {
    std::vector<IntrusivePtr<TestObject>> copies(count);
    for (int i = 0; i < count; ++i) {
      copies[i] = originals[i];
      benchmark::DoNotOptimize(copies[i]);
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_IntrusivePtr_Copy)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SharedPtr_Copy(benchmark::State& state) {
  const int count = state.range(0);
  std::vector<std::shared_ptr<TestObjectShared>> originals(count);
  for (int i = 0; i < count; ++i) { originals[i] = std::make_shared<TestObjectShared>(i); }

  for (auto _ : state) {
    std::vector<std::shared_ptr<TestObjectShared>> copies(count);
    for (int i = 0; i < count; ++i) {
      copies[i] = originals[i];
      benchmark::DoNotOptimize(copies[i]);
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_SharedPtr_Copy)->Arg(100)->Arg(1000)->Arg(10000);

// Benchmark test: move operations
static void BM_IntrusivePtr_Move(benchmark::State& state) {
  const int count = state.range(0);

  for (auto _ : state) {
    std::vector<IntrusivePtr<TestObject>> ptrs;
    ptrs.reserve(count);

    for (int i = 0; i < count; ++i) {
      IntrusivePtr<TestObject> original = make_intrusive<TestObject>(i);
      IntrusivePtr<TestObject> moved = std::move(original);
      ptrs.push_back(std::move(moved));
      benchmark::DoNotOptimize(ptrs.back());
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_IntrusivePtr_Move)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SharedPtr_Move(benchmark::State& state) {
  const int count = state.range(0);

  for (auto _ : state) {
    std::vector<std::shared_ptr<TestObjectShared>> ptrs;
    ptrs.reserve(count);

    for (int i = 0; i < count; ++i) {
      std::shared_ptr<TestObjectShared> original = std::make_shared<TestObjectShared>(i);
      std::shared_ptr<TestObjectShared> moved = std::move(original);
      ptrs.push_back(std::move(moved));
      benchmark::DoNotOptimize(ptrs.back());
    }
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_SharedPtr_Move)->Arg(100)->Arg(1000)->Arg(10000);

// Benchmark test: reference counting operations in multithreaded environment
static void BM_IntrusivePtr_Multithreaded(benchmark::State& state) {
  const int thread_count = state.range(0);
  const int ops_per_thread = state.range(1);

  for (auto _ : state) {
    IntrusivePtr<TestObject> original = make_intrusive<TestObject>(42);
    std::vector<std::thread> threads;
    std::vector<IntrusivePtr<TestObject>> local_ptrs(thread_count * ops_per_thread);

    for (int t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t]() {  // NOLINT
        for (int i = 0; i < ops_per_thread; ++i) {
          local_ptrs[t * ops_per_thread + i] = original;
          local_ptrs[t * ops_per_thread + i]->doWork();
        }
      });
    }

    for (auto& t : threads) { t.join(); }

    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * thread_count * ops_per_thread);
}
BENCHMARK(BM_IntrusivePtr_Multithreaded)
    ->Args({2, 1000})
    ->Args({4, 1000})
    ->Args({8, 1000})
    ->Args({2, 10000})
    ->Args({4, 10000})
    ->Args({8, 10000});

static void BM_SharedPtr_Multithreaded(benchmark::State& state) {
  const int thread_count = state.range(0);
  const int ops_per_thread = state.range(1);

  for (auto _ : state) {
    std::shared_ptr<TestObjectShared> original = std::make_shared<TestObjectShared>(42);
    std::vector<std::thread> threads;
    std::vector<std::shared_ptr<TestObjectShared>> local_ptrs(thread_count * ops_per_thread);

    for (int t = 0; t < thread_count; ++t) {
      threads.emplace_back([&, t]() {  // NOLINT
        for (int i = 0; i < ops_per_thread; ++i) {
          local_ptrs[t * ops_per_thread + i] = original;
          local_ptrs[t * ops_per_thread + i]->doWork();
        }
      });
    }

    for (auto& t : threads) { t.join(); }

    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * thread_count * ops_per_thread);
}
BENCHMARK(BM_SharedPtr_Multithreaded)
    ->Args({2, 1000})
    ->Args({4, 1000})
    ->Args({8, 1000})
    ->Args({2, 10000})
    ->Args({4, 10000})
    ->Args({8, 10000});

// Benchmark test: usage in containers
static void BM_IntrusivePtr_Vector(benchmark::State& state) {
  const int size = state.range(0);
  for (auto _ : state) {
    std::vector<IntrusivePtr<TestObject>> vec;
    vec.reserve(size);

    for (int i = 0; i < size; ++i) { vec.push_back(make_intrusive<TestObject>(i)); }

    // Simulate some container operations
    std::shuffle(vec.begin(), vec.end(), std::mt19937{std::random_device{}()});

    for (auto& ptr : vec) { ptr->doWork(); }

    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_IntrusivePtr_Vector)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SharedPtr_Vector(benchmark::State& state) {
  const int size = state.range(0);
  for (auto _ : state) {
    std::vector<std::shared_ptr<TestObjectShared>> vec;
    vec.reserve(size);

    for (int i = 0; i < size; ++i) { vec.push_back(std::make_shared<TestObjectShared>(i)); }

    // Simulate some container operations
    std::shuffle(vec.begin(), vec.end(), std::mt19937{std::random_device{}()});

    for (auto& ptr : vec) { ptr->doWork(); }

    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_SharedPtr_Vector)->Arg(100)->Arg(1000)->Arg(10000);

// Benchmark test: usage in associative containers
static void BM_IntrusivePtr_Map(benchmark::State& state) {
  const int size = state.range(0);
  for (auto _ : state) {
    std::map<int, IntrusivePtr<TestObject>> obj_map;

    for (int i = 0; i < size; ++i) { obj_map[i] = make_intrusive<TestObject>(i); }

    // Simulate some lookup and update operations
    for (int i = 0; i < size / 10; ++i) {
      int key = i * 10;
      auto it = obj_map.find(key);
      if (it != obj_map.end()) { it->second->doWork(); }
    }

    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_IntrusivePtr_Map)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SharedPtr_Map(benchmark::State& state) {
  const int size = state.range(0);
  for (auto _ : state) {
    std::map<int, std::shared_ptr<TestObjectShared>> obj_map;

    for (int i = 0; i < size; ++i) { obj_map[i] = std::make_shared<TestObjectShared>(i); }

    // Simulate some lookup and update operations
    for (int i = 0; i < size / 10; ++i) {
      int key = i * 10;
      auto it = obj_map.find(key);
      if (it != obj_map.end()) { it->second->doWork(); }
    }

    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * size);
}
BENCHMARK(BM_SharedPtr_Map)->Arg(100)->Arg(1000)->Arg(10000);

// Benchmark test: function passing and returning
static IntrusivePtr<TestObject> processIntrusivePtr(IntrusivePtr<TestObject> ptr) {
  if (ptr) { ptr->doWork(); }
  return ptr;
}

static std::shared_ptr<TestObjectShared> processSharedPtr(std::shared_ptr<TestObjectShared> ptr) {
  if (ptr) { ptr->doWork(); }
  return ptr;
}

static void BM_IntrusivePtr_FunctionPassing(benchmark::State& state) {
  const int count = state.range(0);
  std::vector<IntrusivePtr<TestObject>> ptrs(count);
  for (int i = 0; i < count; ++i) { ptrs[i] = make_intrusive<TestObject>(i); }

  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      auto result = processIntrusivePtr(ptrs[i]);
      benchmark::DoNotOptimize(result);
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_IntrusivePtr_FunctionPassing)->Arg(100)->Arg(1000)->Arg(10000);

static void BM_SharedPtr_FunctionPassing(benchmark::State& state) {
  const int count = state.range(0);
  std::vector<std::shared_ptr<TestObjectShared>> ptrs(count);
  for (int i = 0; i < count; ++i) { ptrs[i] = std::make_shared<TestObjectShared>(i); }

  for (auto _ : state) {
    for (int i = 0; i < count; ++i) {
      auto result = processSharedPtr(ptrs[i]);
      benchmark::DoNotOptimize(result);
    }
  }
  state.SetItemsProcessed(state.iterations() * count);
}
BENCHMARK(BM_SharedPtr_FunctionPassing)->Arg(100)->Arg(1000)->Arg(10000);

BENCHMARK_MAIN();
