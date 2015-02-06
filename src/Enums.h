/*
 * Enums.h
 *
 *  Created on: 24 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_ENUMS_H_
#define SRC_ENUMS_H_

/*
 * Типы границ блока
 */
enum BORDER_TYPE {BY_ANOTHER_BLOCK, BY_FUNCTION};

/*
 * Сторона границы
 */
enum BORDER_SIDE {TOP, LEFT, BOTTOM, RIGHT, BORDER_COUNT};

/*
 * Типы блоков.
 * Центральный процессов или одна их трех видеокарт.
 */
enum BLOCK_TYPE { NULL_BLOCK, CPU, DEVICE0, DEVICE1, DEVICE2 };

int oppositeBorder(int side);
int getDeviceNumber(int blockType);

#endif /* SRC_ENUMS_H_ */
