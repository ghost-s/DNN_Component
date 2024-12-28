#include <iostream>
#include <cuda_runtime.h>
using namespace std;
__global__ void vecadd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
  int n = 10000000;
  float *a, *b, *c;
  int i = 1;
  if(i == 0){
    // cudaMallocManaged是从统一内存分配内存，即主机与设备访问同一块内存区域
    CHECK_CUDA_ERROR(cudaMallocManaged((void**)&a, n * sizeof(float)));
    cudaMallocManaged((void**)&b, n * sizeof(float));
    cudaMallocManaged((void**)&c, n * sizeof(float));
    for (int i = 0; i < n; i++) {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }
    int blocks = ceil(n/256);
    int threads = 256;
    vecadd<<<blocks, threads>>>(a, b, c, n);
  // wait for kernel to finish    CUDA执行是异步的，即设备上的核函数与主机上的程序可能并行执行
    cudaDeviceSynchronize();
    for(int i = 0; i<5; ++i){
      std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }else{
    a = new float[n];
    b = new float[n];
    c = new float[n];

    for (int i = 0; i < n; i++) {
      a[i] = 1.0f;
      b[i] = 2.0f;
    }

    float *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, n * sizeof(float));
    cudaMalloc((void**)&b_d, n * sizeof(float));
    cudaMalloc((void**)&c_d, n * sizeof(float));
    
    cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);
    int blocks = ceil(n/256);
    int threads = 256;
    vecadd<<<blocks, threads>>>(a_d, b_d, c_d, n);
    cudaMemcpy(c, c_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // wait for kernel to finish    CUDA执行是异步的，即设备上的核函数与主机上的程序可能并行执行
    cudaDeviceSynchronize();
    for(int i = 0; i<5; ++i){
      std::cout << a[i] << " " << b[i] << " " << c[i] << std::endl;
    }
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    delete[] a;
    delete[] b;
    delete[] c;
  }
  
  return 0;
}