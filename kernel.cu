
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <helper_cuda.h>
#include <device_functions.h>
#include <helper_timer.h>
#define NUM_BANKS 32
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
int BLOCK_SIZE = 1024;
int SEGMENT_SIZE = 2 * BLOCK_SIZE;

//Define functions...
cudaError_t fullScan(int* out, int* in, const int size); 
cudaError_t fullScan2(int* out, int* in, const int size);

//Prescan with bank conflict avoidance...
__global__ void prescan2(int* outputData, int* inputData, int n, int ss) {
	extern __shared__ int temp[]; //Allocated on invocation - Pointer to shared memory

	//ThreadId - 0 --> total number of threads provided..
	int threadID = threadIdx.x;
	int gThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = 1;

	//Max data access - 2x thread id.
	int ai = gThreadID;
	int bi = gThreadID + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
	if (2 * gThreadID < n) {
		temp[ai + bankOffsetA] = inputData[ai];
		temp[bi + bankOffsetB] = inputData[bi];
	}

	for (int d = ss >> 1; d > 0; d >>= 1) { //Build sum in place up the tree
		__syncthreads();

		if (threadID < d) {
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}


	//Clear the last element
	if (threadID == 0) {
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
	}

	//Traverse the tree and build scan
	for (int d = 1; d < ss; d *= 2) {
		offset >>= 1;
		__syncthreads();

		if (threadID < d) {
			int ai = offset * (2 * threadID + 1) - 1;
			int bi = offset * (2 * threadID + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	//Write the results to the device memory
	if (2 * gThreadID < n) {
		outputData[ai] = temp[ai + bankOffsetA];
		outputData[bi] = temp[bi + bankOffsetB];
	}
}

//Std prescan...
//outputData - output array, inputData - input array, n- arraysize, ss- segment size
__global__ void prescan(int* outputData, int* inputData, int n, int ss) {
	extern __shared__ int temp[]; //Allocated on invocation - Pointer to shared memory

	//ThreadId - 0 --> total number of threads provided..
	int threadID = threadIdx.x;
	int offset = 1;
	int gThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	//Max data access - 2x thread id.
	if (2 * gThreadID < n) {
		temp[2 * threadID] = inputData[2 * gThreadID]; //Load input into shared memory
		temp[2 * threadID + 1] = inputData[2 * gThreadID + 1];
	}

	for (int d = ss>> 1; d > 0; d >>= 1) { //Build sum in place up the tree
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
		temp[ss - 1] = 0;
	}

	//Traverse the tree and build scan
	for (int d = 1; d < ss; d *= 2) {
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
	if (2* gThreadID < n) {
		outputData[2 * gThreadID] = temp[2 * threadID];
		outputData[2 * gThreadID + 1] = temp[2 * threadID + 1];
	}
}



//Main Method
int main()
{
	//Initialise array...k
	const int arraySize = 10000;

	int inputArray[arraySize];
	int inputArray2[arraySize];
	int outputArray[arraySize];
	int outputArray2[arraySize];

	//Create array to input...
	for (int i = 0; i < arraySize; i++) {
		inputArray[i] = 1;
		printf("%d ", inputArray[i]);
	}

	//Create array to input...
	for (int i = 0; i < arraySize; i++) {
		inputArray2[i] = 1;
		printf("%d ", inputArray2[i]);
	}

    // Add vectors in parallel.
    cudaError_t cudaStatus = fullScan(outputArray, inputArray, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Prescan failed!");
        return 1;
    }

	//Output results
	printf("\n\n");
	for(int i = 0; i < arraySize; i++) {
		printf("%d ", outputArray[i]);
	}

	// Add vectors in parallel - Bank conflict Avoidance
	cudaError_t cudaStatus2 = fullScan2(outputArray2, inputArray2, arraySize);
	if (cudaStatus2 != cudaSuccess) {
		fprintf(stderr, "Prescan failed! (bank conflict avoid)");
		return 1;
	}

	//Output results
	printf("\n\n");
	for (int i = 0; i < arraySize; i++) {
		printf("%d ", outputArray2[i]);
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

// Helper function for using CUDA to add vectors in parallel. size - array size;
cudaError_t fullScan(int *out, int* in, const int size)
{
	//Init Stopwatch
	StopWatchInterface* timer = NULL;
	double h_msecs = NULL;

	//Properties for prescan
	int inputVectorSize = size; //Size of the input vector (i.e. array)
	int threadsPerBlock = 1024;
	//int blocksPerGrid = 1 + (inputVectorSize - 1) / threadsPerBlock;
	int blocksPerGrid = ceil(inputVectorSize / (float)SEGMENT_SIZE);
	int sharedMemAmount = (SEGMENT_SIZE) * sizeof(int); //Amount of shared memory given to prescan 

	//Init cuda timer
	cudaEvent_t start, stop;
	float d_msecs;

	//Initialise arrays
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

	//Start Std Timer.
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//Start Cuda Timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Launch prescan.
	printf("\n Launching prescan - \nsize - %d seg size- %d", size, SEGMENT_SIZE);
	printf("\n BlocksPer Grid- %d, threadsPerBlock %d, sharedMemAmount %d", blocksPerGrid, threadsPerBlock, sharedMemAmount);
	prescan <<<blocksPerGrid,threadsPerBlock,sharedMemAmount>>> (dev_out, dev_in,size,SEGMENT_SIZE);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

	//Stop timer.
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer); 
	printf("\n Standard Timer-  Ran in %.5f ms", h_msecs);
	sdkDeleteTimer(&timer);


	//Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_msecs, start, stop);
	printf("\n CUDA Timer-  Ran in %.5f ms", d_msecs);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

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

//FullScan2 - with Bank Conflict Avoidance Prescan...
cudaError_t fullScan2(int* out, int* in, const int size)
{
	//Init Stopwatch
	StopWatchInterface* timer = NULL;
	double h_msecs = NULL;

	//Properties for prescan
	int inputVectorSize = size; //Size of the input vector (i.e. array)
	int threadsPerBlock = 1024;
	//int blocksPerGrid = 1 + (inputVectorSize - 1) / threadsPerBlock;
	int blocksPerGrid = ceil(inputVectorSize / (float)SEGMENT_SIZE);
	int sharedMemAmount = (SEGMENT_SIZE) * sizeof(int); //Amount of shared memory given to prescan 

	//Init cuda timer
	cudaEvent_t start, stop;
	float d_msecs;

	//Initialise arrays
	int* dev_in = 0;
	int* dev_out = 0;

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

	//Start Std Timer.
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	//Start Cuda Timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Launch prescan.
	prescan2 << <blocksPerGrid, threadsPerBlock,sharedMemAmount>>> (dev_out, dev_in, size,SEGMENT_SIZE);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	//Stop timer.
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("\n Bank Conflict - Standard Timer-  Ran in %.5f ms", h_msecs);
	sdkDeleteTimer(&timer);


	//Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_msecs, start, stop);
	printf("\n Bank Conflict - CUDA Timer-  Ran in %.5f ms", d_msecs);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

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

