/*
 * Enums.h
 *
 *  Created on: 24 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_ENUMS_H_
#define SRC_ENUMS_H_

#define BY_FUNCTION -1

#define SAVE_FILE_CODE 253
#define GEOM_FILE_CODE 254
#define VERSION_MAJOR 1
#define VERSION_MINOR 0

enum FLAGS {
	SAVE_FILE = 0x01,
	LOAD_FILE = 0x02,
	TIME_EXECUTION = 0x04,
	STEP_EXECUTION = 0x08,
	STATISTICS = 0x10
};

#define SIZE_INT sizeof(int)
#define SIZE_DOUBLE sizeof(double)
#define SIZE_UN_SH_INT sizeof(unsigned short int)

/*
 * Сторона границы
 */
enum BORDER_SIDE { LEFT, RIGHT, FRONT, BACK, TOP, BOTTOM, BORDER_COUNT };

enum INTERCONNECT_COMPONENT { SIDE, START_X, START_Y, STOP_X, STOP_Y, INTERCONNECT_COMPONENT_COUNT };

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

int getSide(int number);

bool isCPU(int type);
bool isGPU(int type);

#endif /* SRC_ENUMS_H_ */
