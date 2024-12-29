#include <iostream>
#include <cuda_runtime.h>

using namespace std;

int main() {
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);  // 获取设备数量

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 0;
    }
    cout<<"Device Count: "<<deviceCount<<endl;

    int deviceId;
    cudaGetDevice(&deviceId);  // 获取当前设备ID
    
    cout<<"Device ID: "<<deviceId<<endl;

    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, deviceId);

    std::cout << "Max threads per block: " << maxThreadsPerBlock << std::endl;
    

    int maxThreadsDim[3];
    cudaDeviceGetAttribute(&maxThreadsDim[0], cudaDevAttrMaxBlockDimX, deviceId);
    cudaDeviceGetAttribute(&maxThreadsDim[1], cudaDevAttrMaxBlockDimY, deviceId);
    cudaDeviceGetAttribute(&maxThreadsDim[2], cudaDevAttrMaxBlockDimZ, deviceId);

    std::cout << "Max block dimension (x, y, z): "
              << maxThreadsDim[0] << ", "
              << maxThreadsDim[1] << ", "
              << maxThreadsDim[2] << std::endl;

    

    int maxGridDim[3];
    cudaDeviceGetAttribute(&maxGridDim[0], cudaDevAttrMaxGridDimX, deviceId);
    cudaDeviceGetAttribute(&maxGridDim[1], cudaDevAttrMaxGridDimY, deviceId);
    cudaDeviceGetAttribute(&maxGridDim[2], cudaDevAttrMaxGridDimZ, deviceId);

    std::cout << "Max grid dimension (x, y, z): "
              << maxGridDim[0] << ", "
              << maxGridDim[1] << ", "
              << maxGridDim[2] << std::endl;

    cout<<endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);  // 获取设备属性

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions (x, y, z): "
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid dimensions (x, y, z): "
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << std::endl;

    

    return 0;
}
