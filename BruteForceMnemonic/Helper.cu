/**
******************************************************************************
* @author Anton Houzich
* @version V2.0.0 (fixed)
* @date 29-April-2023
* @mail houzich_anton@mail.ru
* discussion https://t.me/brute_force_gpu
******************************************************************************
*/

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Helper.h"

cudaError_t deviceSynchronize(std::string name_kernel) {
    cudaError_t cudaStatus = cudaSuccess;
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetLastError \"%s\" launch failed: %s\n",
            name_kernel.c_str(), cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize \"%s\" returned error code \"%s\"\n",
            name_kernel.c_str(), cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    return cudaStatus;
}

// GPU Architecture SM -> Cores table
// FIX #7: updated to include modern architectures up to SM 9.0 (Hopper)
inline int _ConvertSMVer2Cores(int major, int minor) {
    typedef struct { int SM; int Cores; } sSMtoCores;
    sSMtoCores nGpuArchCoresPerSM[] = {
        { 0x10,   8 }, // Tesla  SM 1.0
        { 0x11,   8 }, // Tesla  SM 1.1
        { 0x12,   8 }, // Tesla  SM 1.2
        { 0x13,   8 }, // Tesla  SM 1.3
        { 0x20,  32 }, // Fermi  SM 2.0
        { 0x21,  48 }, // Fermi  SM 2.1
        { 0x30, 192 }, // Kepler SM 3.0
        { 0x35, 192 }, // Kepler SM 3.5
        { 0x37, 192 }, // Kepler SM 3.7
        { 0x50, 128 }, // Maxwell SM 5.0
        { 0x52, 128 }, // Maxwell SM 5.2
        { 0x53, 128 }, // Maxwell SM 5.3
        { 0x60,  64 }, // Pascal  SM 6.0
        { 0x61, 128 }, // Pascal  SM 6.1
        { 0x62, 128 }, // Pascal  SM 6.2
        { 0x70,  64 }, // Volta   SM 7.0
        { 0x72,  64 }, // Xavier  SM 7.2
        { 0x75,  64 }, // Turing  SM 7.5
        { 0x80,  64 }, // Ampere  SM 8.0
        { 0x86, 128 }, // Ampere  SM 8.6  <-- RTX 3080
        { 0x87, 128 }, // Ampere  SM 8.7
        { 0x89, 128 }, // Ada     SM 8.9
        { 0x90, 128 }, // Hopper  SM 9.0
        { 0x100, 128 }, // Blackwell SM 10.0
        {   -1,  -1 }
    };
    for (int i = 0; i < (int)(sizeof(nGpuArchCoresPerSM)/sizeof(nGpuArchCoresPerSM[0])); i++) {
        if (nGpuArchCoresPerSM[i].SM == -1) break;
        if (nGpuArchCoresPerSM[i].SM == ((major << 4) + minor))
            return nGpuArchCoresPerSM[i].Cores;
    }
    printf("MapSMtoCores for SM %d.%d is undefined. Defaulting to 128 Cores/SM\n", major, minor);
    return 128;
}

void devicesInfo(void) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("\nThere are no available device(s) that support CUDA\n");
    } else {
        printf("\nDetected %d CUDA Capable device(s)\n", deviceCount);
    }

    int driverVersion = 0, runtimeVersion = 0;
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version %d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        char msg[256];
        sprintf(msg, "  Total global memory: %.0f MBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem / 1048576.0f,
            (unsigned long long)deviceProp.totalGlobalMem);
        printf("%s", msg);
        // FIX: clockRate removed from cudaDeviceProp in CUDA 13.x
        printf("  Multiprocessors: %d, CUDA Cores/MP: %d, Total Cores: %d\n",
            deviceProp.multiProcessorCount,
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
            _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
    }
}