/*
 * solver.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#include "solver.h"

Solver::Solver() {
	mCount = 0;
	mState = NULL;
}

Solver::Solver(int _count){
	mCount = _count;
	mState = NULL;
}

void Solver::printMatrix(double* matrix, int zCount, int yCount, int xCount, int cellSize) {
	for (int i = 0; i < zCount; ++i) {
		printf("z = %d\n", i);
		int zShift = xCount * yCount * i;

		for (int j = 0; j < yCount; ++j) {
			int yShift = xCount * j;

			for (int k = 0; k < xCount; ++k) {
				int xShift = k;
				printf("(");
				for (int l = 0; l < cellSize; ++l) {
					int cellShift = l;

					printf("%.2f ", matrix[ (zShift + yShift + xShift)*cellSize + cellShift ]);

				}
				printf(") ");
			}
			printf("\n");
		}
		printf("\n");
	}
}
