/*
 * Enums.h
 *
 *  Created on: 24 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_ENUMS_H_
#define SRC_ENUMS_H_

#define BY_FUNCTION -1

/*#define PERCENTAGE_EXECUTION 1
#define SAVE_FILE 2
#define LOAD_FILE 4*/


enum FLAGS {
	SAVE_FILE = 0x01,
	LOAD_FILE = 0x02,
	TIME_EXECUTION = 0x04,
	STEP_EXECUTION = 0x08
};

/*
 * Сторона границы
 */
enum BORDER_SIDE { TOP, LEFT, BOTTOM, RIGHT, BORDER_COUNT };

/*
 * Типы блоков.
 * Центральный процессов или одна их трех видеокарт.
 */
enum BLOCK_TYPE { NULL_BLOCK, CPU, GPU };

/*
 * Способ выделения памяти.
 */
enum MEMORY_ALLOC_TYPE { NOT_ALLOC, NEW, CUDA_MALLOC, CUDA_MALLOC_HOST };


int oppositeBorder(int side);

bool isCPU(int type);
bool isGPU(int type);

char* blockTypeToString(int type);

#endif /* SRC_ENUMS_H_ */
