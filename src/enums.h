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

#define SOLVER_INIT_STAGE -1

enum FLAGS {
	TIME_EXECUTION = 0x01, LOAD_FILE = 0x02
};

#define SIZE_CHAR sizeof(char)
#define SIZE_INT sizeof(int)
#define SIZE_DOUBLE sizeof(double)
#define SIZE_UN_SH_INT sizeof(unsigned short int)

/*
 * Сторона границы
 */
enum BORDER_SIDE {
	LEFT, RIGHT, FRONT, BACK, TOP, BOTTOM, BORDER_COUNT
};

enum INTERCONNECT_COMPONENT {
	SIDE, M_OFFSET, N_OFFSET, M_LENGTH, N_LENGTH, INTERCONNECT_COMPONENT_COUNT
};

enum SOLVER_TYPE {
	EULER, RK4, DP45
};

enum PROCESSING_UNIT_TYPE {
	CPU_UNIT, GPU_UNIT, NOT_UNIT
};

enum PROBLEM_TYPE {
	ORDINARY, DELAY
};

/*
 * Способ выделения памяти.
 */
enum MEMORY_ALLOC_TYPE {
	NOT_ALLOC, NEW, CUDA_MALLOC, CUDA_MALLOC_HOST
};

/*
 * Статус задачи в базе
 */

enum JOB_STATE {
	JS_NEW, JS_STARTED, JS_PREPROCESSING, JS_QUEUED, JS_RUNNING, JS_CANCELLED, JS_FINISHED, JS_FAILED
};
enum USER_STATUS {
	US_STOP, US_RUN
};

int oppositeBorder(int side);

int getSide(int number);

char* getSideName(int side);

char* getMemoryTypeName(int type);

bool isCPU(int type);
bool isGPU(int type);

#endif /* SRC_ENUMS_H_ */
