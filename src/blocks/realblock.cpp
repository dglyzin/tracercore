/*
 * realblock.cpp
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#include "realblock.h"

RealBlock::RealBlock() {
	// TODO Auto-generated constructor stub

}

RealBlock::~RealBlock() {
	// TODO Auto-generated destructor stub
}

void RealBlock::computeStageBorder(int stage, double time) {
	double* result = problem->getResult(stage, time);
	double* source = problem->getSource(stage, time);

	printf("\nsource must be double**. => source - &source. Error here\n");
	//TODO исправить в ProblemType тип возвращаемого значения для getSource
	pu->computeBorder(mUserFuncs, mCompFuncNumber, result, &source, time, mParams, externalBorder, zCount, yCount, xCount, haloSize);
}

void RealBlock::computeStageCenter(int stage, double time) {
	double* result = problem->getResult(stage, time);
	double* source = problem->getSource(stage, time);

	printf("\nsource must be double**. => source - &source. Error here\n");
	//TODO исправить в ProblemType тип возвращаемого значения для getSource
	pu->computeCenter(mUserFuncs, mCompFuncNumber, result, &source, time, mParams, externalBorder, zCount, yCount, xCount, haloSize);
}

void RealBlock::prepareArgument(int stage, double timestep) {
	problem->prepareArgument(pu, stage, timestep);
}

void RealBlock::prepareStageData(int stage) {

}
