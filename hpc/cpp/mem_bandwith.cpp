#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <vector>

long long test_temporal_simd_w_ms(int K, int N) {
  // long long __attribute__((used)) sum = 0;
  volatile int *arr = new int[N];

  const __m256i zeros = _mm256_set1_epi32(0);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j + 7 < N; j += 8) {
      _mm256_store_si256((__m256i *)&arr[j], zeros);
      // volatile int *p = &arr[j];
      // *p = 42;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto diff_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  delete[] arr;
  return diff_ms;
}

long long test_no_temporal_simd_w_ms(int K, int N) {
  // long long __attribute__((used)) sum = 0;
  volatile int *arr = new int[N];

  const __m256i zeros = _mm256_set1_epi32(0);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j + 7 < N; j += 8) {
      _mm256_stream_si256((__m256i *)&arr[j], zeros);
      // volatile int *p = &arr[j];
      // *p = 42;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto diff_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  delete[] arr;
  return diff_ms;
}

long long test_w_ms(int K, int N) {
  // long long __attribute__((used)) sum = 0;
  int *arr = new int[N];

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      volatile int *p = &arr[j];
      *p = 42;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto diff_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  delete[] arr;
  return diff_ms;
}

long long test_r_ms(int K, int N) {
  // long long __attribute__((used)) sum = 0;
  long long sum = 0;
  int *arr = new int[N];

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
	sum += arr[j];
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  asm volatile("" : : "r,m"(sum) : "memory");
  auto diff_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  delete[] arr;
  return diff_ms;
}

long long test_rw_ms(int K, int N) {
  int *arr = new int[N];
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
        volatile int *p = &arr[j];
        *p = *p + 1;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto diff_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  delete[] arr;
  return diff_ms;
}

void test_rw() 
{
  std::vector<std::pair<long long, long long>> records;
  int K = 2048;
  long long base = 1024;
  std::cout << " RW " << std::endl;
  for (int i = 0; i < 20; ++i) {
    base <<= 1;
    records.emplace_back(base, test_rw_ms(10 + K, base));
    std::cout << std::log2(base * 4) << " \t " << base * (10 + K) << " \t "
              << records.back().second << " \t "
              << (base * (10 + K) / records.back().second * 1000 * 1000 * 1000)
              << std::endl;
    K >>= 1;
  }
}

void test_r() 
{
  std::vector<std::pair<long long, long long>> records;
  int K = 2048;
  long long base = 1024;
  std::cout << " R " << std::endl;
  for (int i = 0; i < 20; ++i) {
    base <<= 1;
    records.emplace_back(base, test_r_ms(10 + K, base));
    std::cout << std::log2(base * 4) << " \t " << base * (10 + K) << " \t "
              << records.back().second << " \t "
              << (base * (10 + K) / records.back().second * 1000 * 1000 * 1000)
              << std::endl;
    K >>= 1;
  }
}

void test_w() {
  std::vector<std::pair<long long, long long>> records;
  int K = 2048;
  long long base = 1024;
  std::cout << " W " << std::endl;
  for (int i = 0; i < 20; ++i) {
    base <<= 1;
    records.emplace_back(base, test_w_ms(10 + K, base));
    std::cout << std::log2(base * 4) << " \t " << base * (10 + K) << " \t "
              << records.back().second << " \t "
              << (base * (10 + K) / records.back().second * 1000 * 1000 * 1000)
              << std::endl;
    K >>= 1;
  }
}

void test_temporal_simd_w() {
  std::vector<std::pair<long long, long long>> records;
  int K = 2048;
  long long base = 1024;
  std::cout << " SIMD W " << std::endl;
  for (int i = 0; i < 20; ++i) {
    base <<= 1;
    records.emplace_back(base, test_temporal_simd_w_ms(10 + K, base));
    std::cout << std::log2(base * 4) << " \t " << base * (10 + K) << " \t "
              << records.back().second << " \t "
              << (base * (10 + K) / records.back().second * 1000 * 1000 * 1000)
              << std::endl;
    K >>= 1;
  }
}

void test_no_temporal_simd_w() {
  std::vector<std::pair<long long, long long>> records;
  int K = 2048;
  long long base = 1024;
  std::cout << " SIMD W " << std::endl;
  for (int i = 0; i < 20; ++i) {
    base <<= 1;
    records.emplace_back(base, test_no_temporal_simd_w_ms(10 + K, base));
    std::cout << std::log2(base * 4) << " \t " << base * (10 + K) << " \t "
              << records.back().second << " \t "
              << (base * (10 + K) / records.back().second * 1000 * 1000 * 1000)
              << std::endl;
    K >>= 1;
  }
}

int main() {
  test_r();
  test_w();
  test_rw();
  test_temporal_simd_w();
  test_no_temporal_simd_w();
  return 0;
}
