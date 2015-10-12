/*
 * block.h
 *
 *  Created on: 12 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_BLOCK_H_
#define SRC_BLOCKS_BLOCK_H_

#include "../problem/ordinary.h"

#include "../processingunit/processingunit.h"

class Block {
public:
	Block();
	virtual ~Block();

	void computeStageBorder(int stage, double time);
	void computeStageCenter(int stage, double time);

	void prepareArgument(int stage, double timestep );

	void prepareStageData(int stage);

protected:
	ProcessingUnit *pc;

	ProblemType* problem;
};

#endif /* SRC_BLOCKS_BLOCK_H_ */
