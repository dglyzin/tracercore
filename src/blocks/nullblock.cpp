/*
 * nullblock.cpp
 *
 *  Created on: 05 нояб. 2015 г.
 *      Author: frolov
 */

#include "../blocks/nullblock.h"

using namespace std;

NullBlock::NullBlock(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _cellSize,
		int _haloSize) :
		Block(_nodeNumber, _dimension, _xCount, _yCount, _zCount, _cellSize, _haloSize) {
}

NullBlock::~NullBlock() {
}

void NullBlock::afterCreate(int problemType, int solverType, double aTol, double rTol) {
	return;
}

void NullBlock::computeStageBorder(int stage, double time) {
	return;
}

void NullBlock::computeStageCenter(int stage, double time) {
	return;
}

void NullBlock::prepareArgument(int stage, double timestep) {
	return;
}

void NullBlock::getSubVolume(double* result, int mStart, int mStop, int nStart, int nStop, int side) {
	return;
}

void NullBlock::setSubVolume(double* source, int mStart, int mStop, int nStart, int nStop, int side) {
	return;
}

void NullBlock::prepareStageData(int stage) {
	return;
}

void NullBlock::prepareStageSourceResult(int stage, double timeStep, double currentTime) {
	return;
}

bool NullBlock::isRealBlock() {
	return false;
}

int NullBlock::getBlockType() {
	return NOT_UNIT;
}

int NullBlock::getDeviceNumber() {
	return -1;
}

bool NullBlock::isBlockType(int type) {
	return false;
}

bool NullBlock::isDeviceNumber(int number) {
	return false;
}

bool NullBlock::isProcessingUnitCPU() {
	return false;
}

bool NullBlock::isProcessingUnitGPU() {
	return false;
}

double NullBlock::getStepError(double timestep) {
	return 0.0;
}

void NullBlock::confirmStep(double timestep) {
	return;
}

void NullBlock::rejectStep(double timestep) {
	return;
}

double* NullBlock::addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) {
	return NULL;
}

double* NullBlock::addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength,
		double* border) {
	return NULL;
}

void NullBlock::moveTempBorderVectorToBorderArray() {
	return;
}

void NullBlock::getCurrentState(double* result) {
	return;
}

void NullBlock::saveStateForDraw(char* path) {
	return;
}

void NullBlock::saveStateForLoad(char* path) {
	return;
}

void NullBlock::saveStateForDrawDenseOutput(char* path, double timestep, double tetha) {
	return;
}

void NullBlock::loadState(ifstream& in) {
	in.seekg(getGridElementCount() * SIZE_DOUBLE, ios::cur);
}

bool NullBlock::isNan() {
	return false;
}

ProcessingUnit* NullBlock::getPU() {
	return NULL;
}

void NullBlock::print() {
	return;
}
