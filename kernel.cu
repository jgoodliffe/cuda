
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <device_functions.h>


cudaError_t addWithCuda(int* out, int* in, const int size);


__global__ void prescan(int* outputData, int* inputData, int n) {
	extern __shared__ int temp[]; //Allocated on invocation - Pointer to shared memory

	//ThreadId - 0 --> total number of threads provided..
	int threadID = threadIdx.x;
	int offset = 1;

	//Max data access - 2x thread id.
	temp[2 * threadID] = inputData[2 * threadID]; //Load input into shared memory
	temp[2 * threadID + 1] = inputData[2 * threadID + 1];

	for (int d = n >> 1; d > 0; d >>= 1) { //Build sum in place up the tree
		__syncthreads();

		if (threadID < d) {
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}


	//Clear the last element
	if (threadID == 0) {
		temp[n - 1] = 0;
	}

	//Traverse the tree and build scan
	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();

		if (threadID < d) {
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	//Write the results to the device memory
	outputData[2 * threadID] = temp[2 * threadID];
	outputData[2 * threadID + 1] = temp[2 * threadID + 1];
}

int main()
{

	//Initialise array...
	const int arraySize = 1024;

	int inputArray[arraySize];
	int outputArray[arraySize];

	//Create array to input...
	for (int i = 0; i < arraySize; i++) {
		inputArray[i] = 1;
		printf("%d ", inputArray[i]);
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(outputArray, inputArray, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Prescan failed!");
        return 1;
    }

	//Output results
	printf("\n\n");
	for(int i = 0; i < arraySize; i++) {
		printf("%d ", outputArray[i]);
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *out, int* in, const int size)
{
    int *dev_in = 0;
    int *dev_out = 0;
    
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy input from system to GPU!");
        goto Error;
	}


	//Launch prescan.
	int BLOCK_SIZE = 256;
	int inputVectorSize = 1; //Size of the input vector (i.e. array)
	int threadsPerBlock = size;
	int blocksPerGrid = ceil(inputVectorSize /(float)BLOCK_SIZE);
	prescan <<<blocksPerGrid,threadsPerBlock,2048*sizeof(int)>>> (dev_out, dev_in, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy from GPU to Host!");
        goto Error;
    }

Error:
    cudaFree(dev_in);
    cudaFree(dev_out);
    
    return cudaStatus;
}
