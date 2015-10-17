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
	double* source = problem->getCurrentStateStageData(stage);
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		double* result = blockBorder[i];

		int index = INTERCONNECT_COMPONENT_COUNT * i;

		/*double* source = NULL;
		cout << endl << "Source = NULL" << endl;*/

		int mStart = sendBorderInfo[ index + M_OFFSET ];
		int mStop = mStart + sendBorderInfo[ index + M_LENGTH ];

		int nStart = sendBorderInfo[ index + N_OFFSET ];
		int nStop = nStart + sendBorderInfo[ index + N_LENGTH ];
		//cout<<"This is block "<<blockNumber<<"preparing data to send: "<< mStart<<" "<<mStop<<" "<<nStart<<" "<<nStop<<endl;
		//cout<< "side is "<<sendBorderInfo[index + SIDE]<<endl;
		switch (sendBorderInfo[index + SIDE]) {
			case LEFT:
				//prepareBorder(i, stage, mStart, mStop, nStart, nStop, 0, haloSize);
				pu->prepareBorder(result, source, mStart, mStop, nStart, nStop, 0, haloSize, yCount, xCount, cellSize);
				break;
			case RIGHT:
				//prepareBorder(i, stage, mStart, mStop, nStart, nStop, xCount - haloSize, xCount);
				pu->prepareBorder(result, source, mStart, mStop, nStart, nStop, xCount - haloSize, xCount, yCount, xCount, cellSize);
				break;
			case FRONT:
				//prepareBorder(i, stage, mStart, mStop, 0, haloSize, nStart, nStop);
				pu->prepareBorder(result, source, mStart, mStop, 0, haloSize, nStart, nStop, yCount, xCount, cellSize);
				break;
			case BACK:
				//prepareBorder(i, stage, mStart, mStop, yCount - haloSize, yCount, nStart, nStop);
				pu->prepareBorder(result, source, mStart, mStop, yCount - haloSize, yCount, nStart, nStop, yCount, xCount, cellSize);
				break;
			case TOP:
				//prepareBorder(i, stage, 0, haloSize, mStart, mStop, nStart, nStop);
				pu->prepareBorder(result, source, 0, haloSize, mStart, mStop, nStart, nStop, yCount, xCount, cellSize);
				break;
			case BOTTOM:
				//prepareBorder(i, stage, zCount - haloSize, zCount, mStart, mStop, nStart, nStop);
				pu->prepareBorder(result, source, zCount - haloSize, zCount, mStart, mStop, nStart, nStop, yCount, xCount, cellSize);
				break;
			default:
				break;
		}
	}
}

bool RealBlock::isRealBlock() {
	return true;
}

int RealBlock::getBlockType() {
	//TODO изменит enums!!!
	return 0;
}

double RealBlock::getStepError(double timestep) {
	return problem->getStepError(pu, timestep);
}

void RealBlock::confirmStep(double timestep) {
	problem->confirmStep(pu, timestep);
}

void RealBlock::rejectStep(double timestep) {
	problem->rejectStep(pu, timestep);
}

