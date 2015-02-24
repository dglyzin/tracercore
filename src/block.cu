/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "block.h"

Block::Block(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber) {
	length = _length;
	width = _width;

	lenghtMove = _lengthMove;
	widthMove = _widthMove;

	nodeNumber = _nodeNumber;

	deviceNumber = _deviceNumber;

	countSendSegmentBorder = countReceiveSegmentBorder = 0;

	sendBorderType = NULL;
	receiveBorderType = NULL;

	blockBorder = NULL;
	externalBorder = NULL;

	blockBorderMove = NULL;
	externalBorderMove = NULL;
	
	blockBorderMemoryAllocType = NULL;
	externalBorderMemoryAllocType = NULL;

	matrix = newMatrix = NULL;
}

Block::~Block() {

}

/*
 * Проверяется, попадает ли сдвиг на границу.
 * Есть возможность выйти за пределы массива границы.
 *
 * Например: длина границы 20, сдвиг 25.
 * Если проверка не выполнена, то тбудет возвращен указател на область, которая по факту не относится к границе.
 * Это будет ошибка.
 *
 * Данная функция также проверяет, что переданный "номер" стороны является корректным.
 */
bool Block::checkValue(int side, int move) {
	if( (side == TOP || side == BOTTOM) && move > width )
		return true;

	if( (side == LEFT || side == RIGHT) && move > length )
		return true;

	if( side >= BORDER_COUNT )
		return true;

	return false;
}

void Block::freeMemory(int memory_alloc_type, double* memory) {
	if(memory == NULL)
		return;
	
	switch(memory_alloc_type) {
		case NOT_ALLOC:
			break;
			
		case NEW:
			delete memory;
			break;
			
		case CUDA_MALLOC:
			cudaFree(memory);
			break;
			
		case CUDA_MALLOC_HOST:
			cudaFreeHost(memory);
			break;
			
		default:
			break;
	}
}

void Block::freeMemory(int memory_alloc_type, int* memory) {
	if(memory == NULL)
		return;
	
	switch(memory_alloc_type) {
		case NOT_ALLOC:
			break;
			
		case NEW:
			delete memory;
			break;
			
		case CUDA_MALLOC:
			cudaFree(memory);
			break;
			
		case CUDA_MALLOC_HOST:
			cudaFreeHost(memory);
			break;
			
		default:
			break;
	}
}