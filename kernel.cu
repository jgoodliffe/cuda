/*
Name: James Goodliffe
Student ID: 1561223
Tasks Achieved:
**Host Block Scan
**Host Full Scan
**Block Scan (Large Vectors)
**Full Scan (Large Vectors)

Times:
- Host Block Scan: 20.600ms
- Host Full Scan: 43.392ms
- Block Scan without BCAO: 1.996ms
- Block Scan with BCAO: --ms (did not get working)
- Full Scan without BCAO: 3.963ms
- Full Scan with BCAO: --ms (did not get working)


Hardware:
Intel Core i7 6700k - 4GHz, 4 Cores, 8 Threads
nVidia GeForce GTX 1080 - 8GB GDDR5

Implementation Information:
I have spent over 48hours working on this implementation.

*/
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
cudaError_t fullScan(int* out, int* in, const int size, int* sums_out, int* cumulative_sums, int* L2_out, int* L3_out); 
cudaError_t fullScan_bcao(int* out, int* in, const int size, int* sums_out, int* cumulative_sums, int* L2_out, int* L3_out);
cudaError_t blockScan(int* out, int* in, const int size);
cudaError_t blockScan_bcao(int* out, int* in, const int size);


//Prescan with bank conflict avoidance...
__global__ void preScan_bcao(int* outputData, int* inputData, int n, int ss) {
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

//Kernels (CUDA Functions)
//outputData - output array, inputData - input array, arraySize- arraysize, segSize- segment size
__global__ void prescan_with_sum(int* outputData, int* inputData, int arraySize, int segSize, int* sums) {
	extern __shared__ int temp[]; //Allocated on invocation - Pointer to shared memory

	//ThreadId - 0 --> total number of threads provided..
	int threadID = threadIdx.x;
	int offset = 1;
	int gThreadID = blockIdx.x * blockDim.x + threadIdx.x;

	//Max data access - 2x thread id.
	if (2 * gThreadID < arraySize) {
		temp[2 * threadID] = inputData[2 * gThreadID]; //Load input into shared memory
		temp[2 * threadID + 1] = inputData[2 * gThreadID + 1];
	}

	for (int d = segSize>> 1; d > 0; d >>= 1) { //Build sum in place up the tree
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
		//Capture 'sum' here...
		sums[blockIdx.x] = temp[segSize - 1];
		//printf("NEW SUM: %d", sums[blockIdx.x]);
		temp[segSize - 1] = 0;
	}

	//Traverse the tree and build scan
	for (int d = 1; d < segSize; d *= 2) {
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
	if (2* gThreadID < arraySize) {
		outputData[2 * gThreadID] = temp[2 * threadID];
		outputData[2 * gThreadID + 1] = temp[2 * threadID + 1];
	}
}

//Basic Prescan without summing.... (to perform a scan on the sums)
__global__ void prescan_without_sum(int* outputData, int* inputData, int arraySize, int segSize) {
	extern __shared__ int temp[]; //Allocated on invocation - Pointer to shared memory

	//ThreadId - 0 --> total number of threads provided..
	int threadID = threadIdx.x;
	int offset = 1;
	int gThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	//Sums array - (gThreadID/BlockSize - 1)

	//Max data access - 2x thread id.
	if (2 * gThreadID < arraySize) {
		temp[2 * threadID] = inputData[2 * gThreadID]; //Load input into shared memory
		temp[2 * threadID + 1] = inputData[2 * gThreadID + 1];
	}

	for (int d = segSize >> 1; d > 0; d >>= 1) { //Build sum in place up the tree
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
		temp[segSize - 1] = 0;
	}

	//Traverse the tree and build scan
	for (int d = 1; d < segSize; d *= 2) {
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
	if (2 * gThreadID < arraySize) {
		outputData[2 * gThreadID] = temp[2 * threadID];
		outputData[2 * gThreadID + 1] = temp[2 * threadID + 1];
	}
}

//Modifies the input array to have all values <blockSize with the SumsArray values... 
__global__ void addSumsToScan(int* summedArray, int* inputArray, int* sumsArray) {
	int gThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	summedArray[gThreadID] = inputArray[gThreadID] + sumsArray[(blockIdx.x/2)];
	//printf("\n addSums: %d",summedArray[gThreadID]);
}


//BlockScan - Host
__host__ void hostBlockScan(int* out, int* in, int size){
	for (int i = 0; i < size; i++){
		out[i] = (i % SEGMENT_SIZE == 0) ? 0 : out[i - 1] + in[i - 1];
	}
}

//FullScan - Host
__host__ void hostFullScan(int* out, int* in, int size){
	for (int i = 0; i < size; i++){
		out[i] = (i == 0) ? 0 : out[i - 1] + in[i - 1];
	}
}

//Compare two arrays.
bool compareArray(int* arr1, int* arr2, int size) {
	for (int i = 0; i < size; i++) {
		//printf("\n arr1 i = %d, arr2 i = %d", arr1[i], arr2[i]);
		if (arr1[i] != arr2[i]) {
			return false;
		} 
	}
	return true;
}

//Main Method
int main()
{
	//Initialise array...k
	const int arraySize = 10000000;
	printf("Running Block Scan for Array Size of %d...", arraySize);

	int sumArraySize = ceil(arraySize / float(SEGMENT_SIZE));

	//Malloc all arrays
	int* sumsArray = (int*)malloc(sumArraySize * sizeof(int));
	int* cumSumsArray = (int*)malloc(sumArraySize * sizeof(int));
	int* inputArray = (int*)malloc(arraySize * sizeof(int));
	int* inputArray2 = (int*)malloc(arraySize * sizeof(int));
	int* outputArray = (int*)malloc(arraySize * sizeof(int));
	int* outputArray2 = (int*)malloc(arraySize * sizeof(int));
	int* l2Array = (int*)malloc(arraySize * sizeof(int));
	int* l3Array = (int*)malloc(arraySize * sizeof(int));
	int* blockScanReturn = (int*)malloc(arraySize * sizeof(int));
	int* hostL3Array = (int*)malloc(arraySize * sizeof(int));
	int* hostBlockScanArray = (int*)malloc(arraySize * sizeof(int));


	//Create array to input...
	for (int i = 0; i < arraySize; i++) {
		inputArray[i] = rand() %10;
		//printf("%d ", inputArray[i]);
	}

	//Create BCAO array to input...
	for (int i = 0; i < arraySize; i++) {
		inputArray2[i] = rand() %10;
		//printf("%d ", inputArray2[i]);
	}

	//Host BlockScan
	StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	hostBlockScan(hostBlockScanArray, inputArray, arraySize);
	sdkStopTimer(&timer);
	double h_msecs = sdkGetTimerValue(&timer);
	printf("\nHost Block Scan time: %fms", h_msecs);
	sdkDeleteTimer(&timer);

	//BlockScan - without BCAO
	printf("\n Running a Block Scan for %d items..", arraySize);
	cudaError_t cudaStatus = blockScan(blockScanReturn, inputArray, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nBlock Scan without BCAO failed!");
		return 1;
	}

	printf("\n Checking Block Scan Result... (without BCAO)");
	if (compareArray(hostBlockScanArray, blockScanReturn, arraySize)) {
		printf("\n Block Scan without BCAO successful.");
	}
	else {
		printf("\n Block Scan without BCAO unsuccessful. Check for errors!");
	}

	//Host FullScan
	StopWatchInterface* timer2 = NULL;
	sdkCreateTimer(&timer2);
	sdkStartTimer(&timer2);
	hostFullScan(hostL3Array, inputArray, arraySize);
	sdkStopTimer(&timer2);
	double h_msecs2 = sdkGetTimerValue(&timer2);
	printf("\nHost Full Scan time: %fms", h_msecs2);
	sdkDeleteTimer(&timer2);

	printf("\nNow running a Full Scan for %d items..", arraySize);
	//FullScan - without BCAO
	cudaError_t cudaStatus2 = fullScan(outputArray, inputArray, arraySize, sumsArray, cumSumsArray,l2Array,l3Array);
	if (cudaStatus2 != cudaSuccess) {
		fprintf(stderr, "\nFull Scan without BCAO failed!");
		return 1;
	}

	//Check Results
	printf("\n Checking Full Scan Result... (without BCAO)");
	if (compareArray(hostL3Array,l3Array,arraySize)) {
		printf("\n Full Scan without BCAO successful.");
	}
	else {
		printf("\n Full Scan without BCAO unsuccessful. Check for errors!");
	}

	//BlockScan - BCAO
	printf("\n Running a Block Scan (with BCAO) for %d items..", arraySize);
	cudaError_t cudaStatus3 = blockScan_bcao(outputArray2, inputArray2, arraySize);
	if (cudaStatus3 != cudaSuccess) {
		fprintf(stderr, "BlockScan with BCAO failed! (bank conflict avoid)");
		return 1;
	}

	//FullScan - BCAO

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\ncudaDeviceReset failed!");
        return 1;
    }

	//Free Memory:
	free(inputArray);
	free(inputArray2);
	free(outputArray);
	free(outputArray2);
	free(sumsArray);
	free(l2Array);
	free(l3Array);
	free(hostBlockScanArray);
	free(hostL3Array);
    return 0;
}

//FullScan - Without BCAO
cudaError_t fullScan(int* out, int* in, const int arraySize, int* sums_out, int* cumulative_sums, int* L2_out, int* L3_out)
{
	//Properties for prescan
	int inputVectorSize = arraySize; //Size of the input vector (i.e. array)
	int threadsPerBlock = 1024;
	//int blocksPerGrid = 1 + (inputVectorSize - 1) / threadsPerBlock;
	int blocksPerGrid = ceil(inputVectorSize / (float)SEGMENT_SIZE);
	int blocksPerGrid_L2 = ceil(inputVectorSize /(float)threadsPerBlock);
	int blocksPerGrid_L2_V2 = ceil(inputVectorSize / (float)(SEGMENT_SIZE * SEGMENT_SIZE));
	int blocksPerGrid_L3 = 1; //(inputVectorSize)/SEGMENT_SIZE^3 = 1
	int sharedMemAmount = (SEGMENT_SIZE) * sizeof(int); //Amount of shared memory given to prescan 

	//Init cuda timer
	cudaEvent_t start, stop;
	float d_msecs;

	//Initialise arrays
    int *l1_in = 0; //Input array
    int *l1_out = 0; //1st Scan Output
	int* dev_sums = 0; //Sums 
 	int* dev_cumulative = 0; //Cumulative sums
	int* l2_out = 0; //2nd Scan Output
	int* dev_L3 = 0; //3rd Scan Output
	int* l3_temp = 0; //Temp Scan Output 1
	int* l3_temp2 = 0; //Temp Scan Output 2
	int* l3_temp3 = 0; //Temp Scan Output 3
	int* l3_temp4 = 0; //Temp Scan Output 3
	int* sums_L3_Stage1 = 0; //L3 Scan Sums - Stage 1
	int* sums_L3_Stage2 = 0; //L3 Scan Sums - Stage 2
	int* dev_cumulative_L2 = 0; // L3 Cumulative Sums

	int dev_sums_size = ceil(arraySize/(float)SEGMENT_SIZE)*sizeof(int);
	int dev_L3_sums_size = blocksPerGrid/(float)SEGMENT_SIZE;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (one input, one output, one sums)

    cudaStatus = cudaMalloc((void**)&l1_in, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&l1_out, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&l2_out, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_L3, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&l3_temp, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&l3_temp2, (blocksPerGrid * sizeof(int)));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&l3_temp3, (blocksPerGrid_L2_V2 * sizeof(int)));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&l3_temp4, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_sums, dev_sums_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&sums_L3_Stage1, (blocksPerGrid*sizeof(int)));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&sums_L3_Stage2, (blocksPerGrid_L2_V2*(sizeof(int))));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&dev_cumulative, dev_sums_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_cumulative_L2, dev_sums_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(l1_in, in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy input from system to GPU!");
        goto Error;
	}

	//Start Cuda Timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Launch Level 1 Scan.
	//printf("\n Launching prescan - \nsize - %d seg size- %d", size, SEGMENT_SIZE);
	//printf("\n BlocksPer Grid- %d, threadsPerBlock %d, sharedMemAmount %d", blocksPerGrid, threadsPerBlock, sharedMemAmount);
	prescan_with_sum <<<blocksPerGrid,threadsPerBlock,sharedMemAmount>>> (l1_out, l1_in,arraySize,SEGMENT_SIZE,dev_sums); 

	//Level 2 - Run a prescan on the sums array.
	prescan_without_sum <<<blocksPerGrid,threadsPerBlock,sharedMemAmount>>> (dev_cumulative,dev_sums,dev_sums_size,SEGMENT_SIZE);
	addSumsToScan << <blocksPerGrid_L2, threadsPerBlock, sharedMemAmount >> > (l2_out, l1_out, dev_cumulative); 

	//Level 3 - 
	//prescan_with_sum(outputData, inputData, arraySize, segSize, sums)
	prescan_with_sum << <blocksPerGrid, threadsPerBlock, sharedMemAmount >> > (l3_temp, l1_in, arraySize, SEGMENT_SIZE, sums_L3_Stage1);
	prescan_with_sum << <blocksPerGrid_L2_V2, threadsPerBlock, sharedMemAmount >> > (l3_temp2, sums_L3_Stage1, blocksPerGrid, SEGMENT_SIZE,sums_L3_Stage2);
	//prescan_without_sum(outputData, inputData, arraySize, segSize)
	prescan_without_sum << <blocksPerGrid_L3, threadsPerBlock, sharedMemAmount >> > (l3_temp3,sums_L3_Stage2,blocksPerGrid_L2_V2,SEGMENT_SIZE);
	//addSumsToScan(summedArray, inputArray, sumsArray)
	addSumsToScan << <blocksPerGrid_L2_V2*2*2,threadsPerBlock>> > (l3_temp4,l3_temp2, l3_temp3);
	addSumsToScan << <blocksPerGrid*2, threadsPerBlock>> > (dev_L3, l3_temp,l3_temp4);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\naddKernel launch failed: %s", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

	//Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_msecs, start, stop);
	printf("\n Full Scan without BCAO -  Time taken:  %.5f ms", d_msecs);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Scan - Copy output array from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, l1_out, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\nFailed to copy from GPU to Host!");
        goto Error;
    }

	// Sums - Copy output sums from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(sums_out, dev_sums, dev_sums_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy from GPU to Host!");
		goto Error;
	}

	// Cumulative - Copy output sums from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(cumulative_sums, dev_cumulative, dev_sums_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy from GPU to Host!");
		goto Error;
	}

	// L2 - Copy output sums from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(L2_out, l2_out , arraySize*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy from GPU to Host!");
		goto Error;
	}

	// L3 - Copy output sums from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(L3_out, dev_L3, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy from GPU to Host!");
		goto Error;
	}

Error:
    cudaFree(l1_in);
    cudaFree(l1_out);
	cudaFree(dev_sums);
	cudaFree(dev_cumulative);
	cudaFree(sums_L3_Stage1);
	cudaFree(sums_L3_Stage2);
	cudaFree(dev_cumulative_L2);
	cudaFree(l2_out);
	cudaFree(dev_L3);
	cudaFree(l3_temp);
	cudaFree(l3_temp2);
	cudaFree(l3_temp3);
	cudaFree(l3_temp4);

    
    return cudaStatus;
}

//FullScan - with BCAO
cudaError_t fullScan_bcao(int* out, int* in, const int size, int* sums_out, int* cumulative_sums, int* L2_out, int* L3_out)
{

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

	//Start Cuda Timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Launch prescan.
	preScan_bcao << <blocksPerGrid, threadsPerBlock,sharedMemAmount>>> (dev_out, dev_in, size,SEGMENT_SIZE);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	//Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_msecs, start, stop);
	printf("\n BCAO Full Scan Ran in %.5f ms", d_msecs);

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

//BlockScan - without BCAO
cudaError_t blockScan(int* out, int* in, const int arraySize)
{
	//Properties for Block Scan
	int inputVectorSize = arraySize; //Size of the input vector (i.e. array)
	int threadsPerBlock = 1024;
	int blocksPerGrid = ceil(inputVectorSize / (float)SEGMENT_SIZE);
	int sharedMemAmount = (SEGMENT_SIZE) * sizeof(int); //Amount of shared memory given to prescan 

	//Init cuda timer
	cudaEvent_t start, stop;
	float d_msecs;

	//Initialise arrays
	int* internalIn = 0; //Input array
	int* internalOut = 0; //Output array

	int dev_sums_size = ceil(arraySize / (float)SEGMENT_SIZE) * sizeof(int);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (one input, one output)    .

	cudaStatus = cudaMalloc((void**)&internalIn, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&internalOut, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(internalIn, in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy input from system to GPU!");
		goto Error;
	}

	//Start Cuda Timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Level 2 - Run a prescan on the sums array.
	prescan_without_sum <<<blocksPerGrid, threadsPerBlock, sharedMemAmount >> > (internalOut,internalIn,arraySize,SEGMENT_SIZE);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	//Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_msecs, start, stop);
	printf("\nBlock Scan (without BCAO)-  Time taken: %.5fms", d_msecs);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// BlockScan - Copy output array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(out, internalOut, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy from GPU to Host!");
		goto Error;
	}

Error:
	cudaFree(internalIn);
	cudaFree(internalOut);
	return cudaStatus;
}

//BlockScan - with BCAO
cudaError_t blockScan_bcao(int* out, int* in, const int arraySize)
{
	//Properties for Block Scan
	int inputVectorSize = arraySize; //Size of the input vector (i.e. array)
	int threadsPerBlock = 1024;
	int blocksPerGrid = ceil(inputVectorSize / (float)SEGMENT_SIZE);
	int sharedMemAmount = (SEGMENT_SIZE) * sizeof(int); //Amount of shared memory given to prescan 

	//Init cuda timer
	cudaEvent_t start, stop;
	float d_msecs;

	//Initialise arrays
	int* internalIn = 0; //Input array
	int* internalOut = 0; //Output array

	int dev_sums_size = ceil(arraySize / (float)SEGMENT_SIZE) * sizeof(int);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for two vectors (one input, one output)    .

	cudaStatus = cudaMalloc((void**)&internalIn, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&internalOut, arraySize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(internalIn, in, arraySize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy input from system to GPU!");
		goto Error;
	}

	//Start Cuda Timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Level 2 - Run a prescan on the sums array.
	prescan_without_sum << <blocksPerGrid, threadsPerBlock, sharedMemAmount >> > (internalOut, internalIn, arraySize, SEGMENT_SIZE);


	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	//Stop CUDA timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&d_msecs, start, stop);
	printf("\n Block Scan with BCAO -  Time taken: %.5f ms", d_msecs);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\ncudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// BlockScan - Copy output array from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(out, internalOut, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\nFailed to copy from GPU to Host!");
		goto Error;
	}

Error:
	cudaFree(internalIn);
	cudaFree(internalOut);
	return cudaStatus;
}
