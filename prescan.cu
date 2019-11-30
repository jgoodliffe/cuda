
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_functions.h>
#include <cuda.h>
#include <stdio.h>

__global__ void prescan(float* g_odata, float* g_idata, int n) {
	extern __shared__ float temp[]; //Allocated on invocation

	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = g_idata[2 * thid]; //Load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1];

	for (int id = n >> 1; d > 0; d >>= 1) { //Build sum in place up the tree
		__syncthreads();

		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}


	//Clear the last element
	if (thid == 0) {
		temp[n - 1] = 0;
	}

	//Traverse the trea and build scan
	for (int d = 1; d < n; n = 2) {
		offset >>= 1;
		__syncthreads();

		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t
		}
	}

	__syncthreads();

	//Write the results to the device memory
	g_odata[2 * thid] = temp[2 * thid];
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
}
