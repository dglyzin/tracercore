#include "cuda_func.h"

/*
 * Функция ядра.
 * Расчет теплоемкости на видеокарте.
 * Логика функции аналогична функции для центрального процессора.
 */
/*__global__ void calc ( double* matrix, double* newMatrix, int length, int width, double dX2, double dY2, double dT, int **recieveBorderType, double** externalBorder, int* externalBorderMove ) {
	double top, left, bottom, right, cur;

	int i = BLOCK_LENGHT_SIZE * blockIdx.x + threadIdx.x;
	int j = BLOCK_WIDTH_SIZE * blockIdx.y + threadIdx.y;

	if( i < length && j < width ) {
		if( i == 0 )
			if( recieveBorderType[TOP][j] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 100;
				return;
			}
			else
				top = externalBorder[	recieveBorderType[TOP][j]	][j - externalBorderMove[	recieveBorderType[TOP][j]	]];
		else
			top = matrix[(i - 1) * width + j];
	
	
		if( j == 0 )
			if( recieveBorderType[LEFT][i] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 10;
				return;
			}
			else
				left = externalBorder[	recieveBorderType[LEFT][i]	][i - externalBorderMove[	recieveBorderType[LEFT][i]		]];
		else
			left = matrix[i * width + (j - 1)];
	
	
		if( i == length - 1 )
			if( recieveBorderType[BOTTOM][j] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 10;
				return;
			}
			else
				bottom = externalBorder[	recieveBorderType[BOTTOM][j]	][j - externalBorderMove[	recieveBorderType[BOTTOM][j]	]];
		else
			bottom = matrix[(i + 1) * width + j];
	
	
		if( j == width - 1 )
			if( recieveBorderType[RIGHT][i] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 10;
				return;
			}
			else
				right = externalBorder[	recieveBorderType[RIGHT][i]	][i - externalBorderMove[	recieveBorderType[RIGHT][i]	]];
		else
			right = matrix[i * width + (j + 1)];

	
		cur = matrix[i * width + j];
	
		newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
	}
}*/

/*__global__ void calcBorder ( double* matrix, double* newMatrix, int length, int width, double dX2, double dY2, double dT, int **recieveBorderType, double** externalBorder, int* externalBorderMove ) {
	double top, left, bottom, right, cur;

	int i = BLOCK_LENGHT_SIZE * blockIdx.x + threadIdx.x;
	int j = BLOCK_WIDTH_SIZE * blockIdx.y + threadIdx.y;

	if( i < length && j < width )
		if( i == 0 || i == length - 1 || j == 0 || j == width - 1 ) {
			if( i == 0 )
				if( recieveBorderType[TOP][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 100;
					return;
				}
				else
					top = externalBorder[	recieveBorderType[TOP][j]	][j - externalBorderMove[	recieveBorderType[TOP][j]	]];
			else
				top = matrix[(i - 1) * width + j];
		
		
			if( j == 0 )
				if( recieveBorderType[LEFT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					return;
				}
				else
					left = externalBorder[	recieveBorderType[LEFT][i]	][i - externalBorderMove[	recieveBorderType[LEFT][i]		]];
			else
				left = matrix[i * width + (j - 1)];
		
		
			if( i == length - 1 )
				if( recieveBorderType[BOTTOM][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					return;
				}
				else
					bottom = externalBorder[	recieveBorderType[BOTTOM][j]	][j - externalBorderMove[	recieveBorderType[BOTTOM][j]	]];
			else
				bottom = matrix[(i + 1) * width + j];
		
		
			if( j == width - 1 )
				if( recieveBorderType[RIGHT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					return;
				}
				else
					right = externalBorder[	recieveBorderType[RIGHT][i]	][i - externalBorderMove[	recieveBorderType[RIGHT][i]	]];
			else
				right = matrix[i * width + (j + 1)];
		
		
			cur = matrix[i * width + j];
		
			newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
}*/

/*__global__ void calcCenter ( double* matrix, double* newMatrix, int length, int width, double dX2, double dY2, double dT, int **recieveBorderType, double** externalBorder, int* externalBorderMove ) {
	double top, left, bottom, right, cur;

	int i = BLOCK_LENGHT_SIZE * blockIdx.x + threadIdx.x;
	int j = BLOCK_WIDTH_SIZE * blockIdx.y + threadIdx.y;
	
	if( (i > 1) && (i < length - 1) && (j > 1) && (j < width - 1) ) {
		top = matrix[(i - 1) * width + j];
		left = matrix[i * width + (j - 1)];
		bottom = matrix[(i + 1) * width + j];
		right = matrix[i * width + (j + 1)];

		cur = matrix[i * width + j];
	
		newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
	}
	
	if( i < length && j < width )
		if( i != 0 && i != length - 1 && j != 0 && j != width - 1 ) {
			top = matrix[(i - 1) * width + j];
			left = matrix[i * width + (j - 1)];
			bottom = matrix[(i + 1) * width + j];
			right = matrix[i * width + (j + 1)];
	
			cur = matrix[i * width + j];
		
			newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
}*/

__global__ void assignIntArray (int* array, int value, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		array[idx] = value;
}

__global__ void copyIntArray (int* dest, int* source, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		dest[idx] = source[idx];
}

__global__ void assignDoubleArray (double* array, double value, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		array[idx] = value;
}

__global__ void copyDoubleArray (double* dest, double* source, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		dest[idx] = source[idx];
}

__global__ void sumDoubleArray (double* arg1, double* arg2, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = arg1[idx] + arg2[idx];
}

__global__ void multiplyDoubleArrayByNumber (double* array, double value, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = array[idx] * value;
}



__global__ void multiplyByNumberAndSumDoubleArrays(double* array1, double value1, double* array2, double value2, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = value1 * array1[idx] + value2 * array2[idx];
}

__global__ void multiplyByNumberAndSumDoubleArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = value1 * array1[idx] + value2 * array2[idx] + value3 * array3[idx];
}

__global__ void multiplyByNumberAndSumDoubleArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = value1 * array1[idx] + value2 * array2[idx] + value3 * array3[idx] + value4 * array4[idx];
}

__global__ void multiplyByNumberAndSumDoubleArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* array5, double value5, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = value1 * array1[idx] + value2 * array2[idx] + value3 * array3[idx] + value4 * array4[idx] + value5 * array5[idx];
}

__global__ void multiplyByNumberAndSumDoubleArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* array5, double value5, double* array6, double value6, double* result, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		result[idx] = value1 * array1[idx] + value2 * array2[idx] + value3 * array3[idx] + value4 * array4[idx] + value5 * array5[idx] + value6 * array6[idx];
}


__global__ void sumElementOfDoubleArray(double* array, double* result, int arrayLength) {
    __shared__ double data[BLOCK_SIZE];
    
    int tid=threadIdx.x; 
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    
    data[tid] = ( idx < arrayLength ) ? array[idx] : 0;

    __syncthreads();// ждем пока все нити(потоки) скопируют данные. 
 
    for(int s = blockDim.x/2; s > 0; s = s/2) { 
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



__global__ void prepareBorderDevice(double* dest, double* source, int borderNumber, int zCount, int yCount, int xCount) {
	
}


void assignArray(int* array, int value, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	assignIntArray <<< blocks, threads >>> ( array, value, arrayLength);
}

void assignArray(double* array, double value, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	assignDoubleArray <<< blocks, threads >>> ( array, value, arrayLength);
}

void copyArray(int* dest, int* source, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	copyIntArray <<< blocks, threads >>> ( dest, source, arrayLength);
}


void copyArray(double* dest, double* source, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	copyDoubleArray <<< blocks, threads >>> ( dest, source, arrayLength);
}

void sumArray(double* arg1, double* arg2, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	sumDoubleArray <<< blocks, threads >>> ( arg1, arg2, result, arrayLength);
}

void multiplyArrayByNumber(double* array, double value, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyDoubleArrayByNumber <<< blocks, threads >>> ( array, value, result, arrayLength);
}


void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyByNumberAndSumDoubleArrays <<< blocks, threads >>> ( array1, value1, array2, value2, result, arrayLength);
}

void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyByNumberAndSumDoubleArrays <<< blocks, threads >>> ( array1, value1, array2, value2, array3, value3, result, arrayLength);
}

void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyByNumberAndSumDoubleArrays <<< blocks, threads >>> ( array1, value1, array2, value2, array3, value3, array4, value4, result, arrayLength);
}

void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* array5, double value5, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyByNumberAndSumDoubleArrays <<< blocks, threads >>> ( array1, value1, array2, value2, array3, value3, array4, value4, array5, value5, result, arrayLength);
}

void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* array5, double value5, double* array6, double value6, double* result, int arrayLength) {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
	
	multiplyByNumberAndSumDoubleArrays <<< blocks, threads >>> ( array1, value1, array2, value2, array3, value3, array4, value4, array5, value5, array6, value6, result, arrayLength);
}




void prepareBorder(double* dest, double* source, int borderNumber, int zCount, int yCount, int xCount) {
	printf("\nPreapre border GPU\n");
}

void computeCenter() {
	printf("\nCompute center GPU\n");
}

void computeBorder() {
	printf("\nCompute border GPU\n");
}

double sumElementOfArray(double* array, int arrayLength) {
	double sumHost;
	double* sumDevice;
	
	cudaMalloc( (void**)&sumDevice, 1 * sizeof(double) );
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)arrayLength / threads.x) );
		
	sumElementOfDoubleArray <<< blocks, threads >>> ( array, sumDevice, arrayLength );
	
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