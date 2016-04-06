#include "cuda_func.h"

__global__ void copyArrayCuda (double* source, double* destination, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		destination[idx] = source[idx];
}

__global__ void copyArrayCuda (unsigned short int* source, unsigned short int* destination, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		destination[idx] = source[idx];
}

__global__ void sumArrayCuda (double* result, double* arg1, double* arg2, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = arg1[idx] + arg2[idx];
}

__global__ void multiplyArrayByNumberCuda (double* result, double* arg, double factor, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = arg[idx] * factor;
}

__global__ void multiplyArrayByNumberAndSumCuda(double* result, double* arg1, double factor, double* arg2, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = factor * arg1[idx] + arg2[idx];
}

__global__ void sumArrayElementsCuda(double* array, double* result, int size) {
    __shared__ double data[BLOCK_SIZE];
    
    int tid=threadIdx.x; 
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    data[tid] = ( idx < size ) ? array[idx] : 0;

    // ждем пока все нити(потоки) скопируют данные
    __syncthreads();
 
    for(int s = blockDim.x / 2; s > 0; s = s / 2) { 
        if (tid < s)
        	data[tid] += data[ tid + s ]; 
        __syncthreads(); 
    }
    
    if ( tid==0 ) 
        result[blockIdx.x] = data[0]; 
}

__global__ void maxElementsElementwiseCuda(double* result, double* arg1, double* arg2, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = max( arg1[idx], arg2[idx] );
}

__global__ void divisionArraysElementwiseCuda(double* result, double* arg1, double* arg2, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = arg1[idx] / arg2[idx];
}

__global__ void addNumberToArrayCuda(double* result, double* arg, double number, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = arg[idx] + number;
}

__global__ void multiplyArraysElementwiseCuda(double* result, double* arg1, double* arg2, int size) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < size )
		result[idx] = arg1[idx] * arg2[idx];
}

__global__ void isNanCuda(double* array, bool* result, int size) {
    __shared__ bool data[BLOCK_SIZE];
    
    int tid=threadIdx.x; 
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    data[tid] = ( idx < size ) ? isnan(array[idx]) : 0;

    // ждем пока все нити(потоки) скопируют данные
    __syncthreads();
 
    for(int s = blockDim.x / 2; s > 0; s = s / 2) { 
        if (tid < s)
        	data[tid] |= data[ tid + s ]; 
        __syncthreads(); 
    }
    
    if ( tid==0 ) 
        result[blockIdx.x] = data[0];
}

__global__ void forGetStepErrorDP45(double* mTempStore1, double e1,
		double* mTempStore3, double e3, double* mTempStore4, double e4,
		double* mTempStore5, double e5, double* mTempStore6, double e6,
		double* mTempStore7, double e7, double* mState, double* mArg,
		double timeStep, double aTol, double rTol, double mCount, double* result) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
		
	if( idx < mCount ) {
		
		mTempStore1[idx] = timeStep * (e1 * mTempStore1[idx] + e3 * mTempStore3[idx] + e4 * mTempStore4[idx] + e5 * mTempStore5[idx] + e6 * mTempStore6[idx]+ e7 * mTempStore7[idx]) /
				(aTol + rTol * max(mArg[idx], mState[idx]));
				
		result[idx] = mTempStore1[idx] * mTempStore1[idx];		
	}
}



__global__ void prepareBorderDevice(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {	
	int index = 0;
	for (int z = zStart; z < zStop; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = yStart; y < yStop; ++y) {
			int yShift = xCount * y;

			for (int x = xStart; x < xStop; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;
					//printf("block %d is preparing border %d, x=%d, y=%d, z=%d, index=%d\n", blockNumber, borderNumber, x,y,z, index);

					result[index] = source[(zShift + yShift + xShift) * cellSize + cellShift];
					index++;
				}
			}
		}
	}
}



void copyArrayGPU(double* source, double* destination, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
	
	copyArrayCuda <<< blocks, threads >>> ( source, destination, size);
}

void copyArrayGPU(unsigned short int* source, unsigned short int* destination, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
	
	copyArrayCuda <<< blocks, threads >>> ( source, destination, size);
}

void sumArraysGPU(double* result, double* arg1, double* arg2, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
	
	sumArrayCuda <<< blocks, threads >>> ( result, arg1, arg2, size);
}

void multiplyArrayByNumberGPU(double* result, double* arg, double factor, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
	
	multiplyArrayByNumberCuda <<< blocks, threads >>> ( result, arg, factor, size);
}

void multiplyArrayByNumberAndSumGPU(double* result, double* arg1, double factor, double* arg2, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
	
	multiplyArrayByNumberAndSumCuda <<< blocks, threads >>> ( result, arg1, factor, arg2, size);
}


double sumArrayElementsGPU(double* arg, int size) {
	double sumHost;
	double* sumDevice;
	
	cudaMalloc( (void**)&sumDevice, 1 * sizeof(double) );
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
		
	sumArrayElementsCuda <<< blocks, threads >>> ( arg, sumDevice, size );
	
	cudaMemcpy(&sumHost, sumDevice, 1 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(sumDevice);
	
	return sumHost;
}

void maxElementsElementwiseGPU(double* result, double* arg1, double* arg2, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
		
	maxElementsElementwiseCuda <<< blocks, threads >>> ( result, arg1, arg2, size);
}

void divisionArraysElementwiseGPU(double* result, double* arg1, double* arg2, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
		
	divisionArraysElementwiseCuda <<< blocks, threads >>> ( result, arg1, arg2, size);
}

void addNumberToArrayGPU(double* result, double* arg, double number, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
		
	addNumberToArrayCuda <<< blocks, threads >>> ( result, arg, number, size);
}

void multiplyArraysElementwiseGPU(double* result, double* arg1, double* arg2, int size) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
		
	multiplyArraysElementwiseCuda <<< blocks, threads >>> ( result, arg1, arg2, size);
}

bool isNanGPU(double* array, int size) {
	bool isNanHost;
	bool* isNanDevice;
	
	cudaMalloc( (void**)&isNanDevice, 1 * sizeof(bool) );
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)size / threads.x) );
		
	isNanCuda <<< blocks, threads >>> ( array, isNanDevice, size );
	
	cudaMemcpy(&isNanHost, isNanDevice, 1 * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaFree(isNanDevice);
	
	return isNanHost;
}


void prepareBorderGPU(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {
	prepareBorderDevice <<< 1, 1 >>> (result, source, zStart, zStop, yStart, yStop, xStart, xStop, yCount, xCount, cellSize);
}

void computeCenterGPU_1d() {
	printf("\nCompute center GPU\n");
}

void computeBorderGPU_1d() {
	printf("\nCompute border GPU\n");
}