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

/*
 * Функция ядра
 * Заполнение целочисленного массива определенным значением.
 */
__global__ void assignIntArray (int* arr, int value, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		arr[idx] = value;
}

/*
 * Функция ядра
 * Копирование целочесленных массивов.
 */
__global__ void copyIntArray (int* dest, int* source, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		dest[idx] = source[idx];
}

/*
 * Функция ядра
 * Заполнение вещественного массива определенным значением.
 */
__global__ void assignDoubleArray (double* arr, double value, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		arr[idx] = value;
}

/*
 * Функция ядра
 * Копирование вещественных массивов.
 */
__global__ void copyDoubleArray (double* dest, double* source, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		dest[idx] = source[idx];
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