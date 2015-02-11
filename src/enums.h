/*
 * Enums.h
 *
 *  Created on: 24 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_ENUMS_H_
#define SRC_ENUMS_H_

#define BY_FUNCTION -1

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

bool isCPU(int type);
bool isGPU(int type);

#endif /* SRC_ENUMS_H_ */
