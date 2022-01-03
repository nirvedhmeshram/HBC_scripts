#include <cub/cub.cuh>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
const int SIZE = 5000;
void setupInitialValues(int *offsets, int *lengths) {
  for (int i = 0; i < SIZE; i++) {
    offsets[i] = 0;
    lengths[i] = 1;
  }
}
void printValues(int *offsets, int *lengths) {
  printf("Length values are: ");
  for (int i = 0; i < SIZE; i++) {
    printf("%d ", lengths[i]);
  }
  printf("\n");
  printf("offsets values are: ");
  for (int i = 0; i < SIZE; i++) {
    printf("%d ", offsets[i]);
  }
  printf("\n");
}
void naiveCumSum(int *offsets, int *lengths) {
  offsets[0] = lengths[0];
  for (int i = 1; i < SIZE; i++) {
    offsets[i] = offsets[i - 1] + lengths[i];
  }
}
int main() {
  clock_t start, end;
  double cpu_time_used;
  // start=clock();
  int *offsets, *lengths;
  cudaMallocManaged(&offsets, SIZE * sizeof(int)); // Allocate CPU/GPU Memory
  cudaMallocManaged(&lengths, SIZE * sizeof(int));
  // setup
  setupInitialValues(offsets, lengths);
  //printValues(offsets, lengths);
  start = clock();
  naiveCumSum(offsets, lengths);
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU Time %lf\n", cpu_time_used);
  //printValues(offsets, lengths);
  // This was warmup
  // This is the part we want in codgen for cumsum
  start = clock();
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes;
  cub::DeviceScan::InclusiveSum(
      d_temp_storage, temp_storage_bytes, lengths, offsets,
      SIZE); // This returns the number of bytes for temp_storage
  cudaMallocManaged(&d_temp_storage, temp_storage_bytes);
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, lengths,
                                offsets, SIZE);
  cudaDeviceSynchronize();
  end = clock();
  cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CUB Time %lf\n", cpu_time_used);
  //printValues(offsets, lengths);
  return 0;
}
