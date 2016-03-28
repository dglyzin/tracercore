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
	
	if( idx < arrayLength )
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



__global__ void prepareBorderDevice(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop, double** blockBorder, int zCount, int yCount, int xCount, int cellSize) {	
	/*int zShift;
	int yShift;
	int xShift;
	
	int z;
	int y;
	int x;
	
	int index;
	
	switch (side) {
		case LEFT:
			for (int x = xStart; x < xStop; ++x) {
				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
	}*/
	
	int index = 0;
	for (int z = zStart; z < zStop; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = yStart; y < yStop; ++y) {
			int yShift = xCount * y;

			for (int x = xStart; x < xStop; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
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
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyArrayByNumberAndSumCuda <<< blocks, threads >>> ( result, arg1, factor, arg2, size);
}


double sumArrayElementsGPU(double* arg, int size); {
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

double getStepErrorDP45(double* mTempStore1, double e1,
		double* mTempStore3, double e3, double* mTempStore4, double e4,
		double* mTempStore5, double e5, double* mTempStore6, double e6,
		double* mTempStore7, double e7, double* mState, double* mArg,
		double timeStep, double aTol, double rTol, double mCount) {
	
	double errorHost;
	double* errorDevice;
	
	cudaMalloc( (void**)&errorDevice, 1 * sizeof(double) );
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)mCount / threads.x) );
		
	forGetStepErrorDP45 <<< blocks, threads >>> ( mTempStore1, e1, mTempStore3, e3, mTempStore4, e4, mTempStore5, e5, mTempStore6, e6, mTempStore7, e7, mState, mArg, timeStep, aTol, rTol, mCount, mTempStore1 );
	sumElementOfDoubleArray <<< blocks, threads >>> ( mTempStore1, errorDevice, mCount );
	
	cudaMemcpy(&errorHost, errorDevice, 1 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(errorDevice);
	
	return errorHost;
}

void prepareBorderCudaFunc(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop, double** blockBorder, int zCount, int yCount, int xCount, int cellSize) {
	prepareBorderDevice <<< 1, 1 >>> (source, borderNumber, zStart, zStop, yStart, yStop, xStart, xStop, blockBorder, zCount, yCount, xCount, cellSize);
}

void computeCenter() {
	printf("\nCompute center GPU\n");
}

void computeBorder() {
	printf("\nCompute border GPU\n");
}

/*
 * Функция ядра
 * Копирование данных из матрицы в массив.
 * Используется при подготовке пересылаемых данных.
 */
/*__global__ void copyBorderFromMatrix ( double** blockBorder, double* matrix, int** sendBorderType, int* blockBorderMove, int side, int length, int width ) {
	int idx  = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( (side == TOP || side == BOTTOM) && idx >= width )
		return;
	
	if( (side == LEFT || side == RIGHT) && idx >= length )
		return;

	if( sendBorderType[side][idx] == BY_FUNCTION )
		return;
	
	double value;
	
	switch (side) {
		case TOP:
			value = matrix[0 * width + idx];
			break;
		case LEFT:
			value = matrix[idx * width + 0];
			break;
		case BOTTOM:
			value = matrix[(length - 1) * width + idx];
			break;
		case RIGHT:
			value = matrix[idx * width + (width - 1)];
			break;
		default:
			break;
	}
	
	blockBorder[	sendBorderType[side][idx]	][idx - blockBorderMove[	sendBorderType[side][idx]	]] = value;
}*/