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

	matrix = newMatrix = NULL;
}

Block::~Block() {

}

double* Block::getBorderBlockData(int side, int move) {
	/*
	 * Если входные данные является некорректными, то работа будет завершена.
	 */
	if( checkValue(side, move) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	return blockBorder != NULL ? blockBorder[side] + move : NULL;
}

double* Block::getExternalBorderData(int side, int move) {
	if( checkValue(side, move) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	return externalBorder != NULL ? externalBorder[side] + move : NULL;
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
