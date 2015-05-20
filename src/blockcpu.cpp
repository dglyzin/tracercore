/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "blockcpu.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "userfuncs.h"

using namespace std;

BlockCpu::BlockCpu(int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize,
		unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
		int _solverIndex) :
				Block( _dimension, _xCount, _yCount, _zCount,
				_xOffset, _yOffset, _zOffset,
				_nodeNumber, _deviceNumber,
				_haloSize, _cellSize ) {
	cout << "Creating block..\n";

	createSolver(_solverIndex );


	int count = getGridNodeCount();

	mCompFuncNumber = new unsigned short int [count];

	for (int i = 0; i < count; ++i) {
		mCompFuncNumber[i] = _compFuncNumber[i];
	}

	getFuncArray(&mUserFuncs);
	getInitFuncArray(&mUserInitFuncs);
	initDefaultParams(&mParams, &mParamsCount);


	cout << "Default params ("<<mParamsCount<<"): ";
	for (int idx=0;idx<mParamsCount; idx++)
		cout <<mParams[idx] << " ";
	cout << endl;

	cout << "functions loaded\n";
	//printf("Func array points to %d \n", (long unsigned int) mUserFuncs );

	//mUserFuncs[0](newMatrix, matrix, 0.0, 2, 2, 0, mParams, NULL);
	//printf("Func array points to %d \n", (long unsigned int) mUserInitFuncs );
	double* matrix = mSolver->getStatePtr();
	mUserInitFuncs[0](matrix,_initFuncNumber);
	cout << "Initial values filled \n";

	for (int i = 0; i < zCount; ++i) {
			int zShift = xCount * yCount * i;

			for (int j = 0; j < yCount; ++j) {
				int yShift = xCount * j;

				for (int k = 0; k < xCount; ++k) {
					int xShift = k;
                    printf("(");
					for (int l = 0; l < cellSize; ++l) {
						int cellShift = l;

						printf("%f ", matrix[ (zShift + yShift + xShift)*cellSize + cellShift ]);

					}
					printf(") ");
				}
				printf("\n");
			}
			printf("\n");
		}
}

BlockCpu::~BlockCpu() {
	cout<<"Deleting block"<<endl;
	releaseParams(mParams);
	releaseFuncArray(mUserFuncs);
	releaseInitFuncArray(mUserInitFuncs);
	delete mSolver;
	
	if(blockBorder != NULL) {
		for(int i = 0; i < countSendSegmentBorder; i++ )
			freeMemory(blockBorderMemoryAllocType[i], blockBorder[i]);
		
		delete blockBorder;
		delete blockBorderMemoryAllocType;
	}
	
	
	if(externalBorder != NULL) {
		for(int i = 0; i < countReceiveSegmentBorder; i++ )
			freeMemory(externalBorderMemoryAllocType[i], externalBorder[i]);
		
		delete externalBorder;
		delete externalBorderMemoryAllocType;
	}
}

void BlockCpu::computeStageCenter_1d(int stage, double time, double step) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);
# pragma omp for
		for (int x = haloSize; x < xCount - haloSize; ++x) {
			//cout << "Calc x_" << x << endl;
			mUserFuncs[ mCompFuncNumber[x] ](result, source, time, x, 0, 0, mParams, externalBorder);
		}
	}
}

void BlockCpu::computeStageCenter_2d(int stage, double time, double step) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);
# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = haloSize; x < xCount - haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source, time, x, y, 0, mParams, externalBorder);
			}
		}
	}
}

void BlockCpu::computeStageCenter_3d(int stage, double time, double step) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);
# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = haloSize; x < xCount - haloSize; ++x) {
					int xShift = x;
					//cout << "Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}
	}
}

void BlockCpu::computeStageBorder_1d(int stage, double time, double step) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);
# pragma omp for
		for (int x = 0; x < haloSize; ++x) {
			//cout << "Border Calc x_" << x << endl;
			mUserFuncs[ mCompFuncNumber[x] ](result, source, time, x, 0, 0, mParams, externalBorder);
		}

# pragma omp for
		for (int x = xCount - haloSize; x < xCount; ++x) {
			//cout << "Border Calc x_" << x << endl;
			mUserFuncs[ mCompFuncNumber[x] ](result, source, time, x, 0, 0, mParams, externalBorder);
		}
	}
}

void BlockCpu::computeStageBorder_2d(int stage, double time, double step) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);
# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = 0; y < haloSize; ++y) {
				int yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source, time, x, y, 0, mParams, externalBorder);
			}
		}

# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				int yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source, time, x, y, 0, mParams, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = 0; x < haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source, time, x, y, 0, mParams, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = xCount - haloSize; x < xCount; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source, time, x, y, 0, mParams, externalBorder);
			}
		}
	}
}

void BlockCpu::computeStageBorder_3d(int stage, double time, double step) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);

		for (int z = 0; z < haloSize; ++z) {
			int zShift = yCount * xCount * z;
	# pragma omp for
			for (int y = 0; y < yCount; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

		for (int z = zCount - haloSize; z < zCount; ++z) {
			int zShift = yCount * xCount * z;
	# pragma omp for
			for (int y = 0; y < yCount; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = 0; y < haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < haloSize; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = xCount - haloSize; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}
	}
}

/*void BlockCpu::prepareStageData(int stage) {
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;

		double* source = NULL;
		cout << endl << "Source = NULL" << endl;

		int mStart = sendBorderInfo[ index + M_OFFSET ];
		int mStop = mStart + sendBorderInfo[ index + M_LENGTH ];

		int nStart = sendBorderInfo[ index + N_OFFSET ];
		int nStop = nStart + sendBorderInfo[ index + N_LENGTH ];

		switch (sendBorderInfo[index + SIDE]) {
			case LEFT:
				prepareBorder(source, i, mStart, mStop, nStart, nStop, 0, haloSize);
				break;
			case RIGHT:
				prepareBorder(source, i, mStart, mStop, nStart, nStop, xCount - haloSize, xCount);
				break;
			case FRONT:
				prepareBorder(source, i, mStart, mStop, 0, haloSize, nStart, nStop);
				break;
			case BACK:
				prepareBorder(source, i, mStart, mStop, yCount - haloSize, yCount, nStart, nStop);
				break;
			case TOP:
				prepareBorder(source, i, 0, haloSize, mStart, mStop, nStart, nStop);
				break;
			case BOTTOM:
				prepareBorder(source, i, zCount - haloSize, zCount, mStart, mStop, nStart, nStop);
				break;
			default:
				break;
		}
	}
}*/

double BlockCpu::getSolverStepError(double timeStep, double aTol, double rTol) {
	return mSolver->getStepError(timeStep, aTol, rTol);
}


void BlockCpu::getCurrentState(double* result) {
	mSolver->copyState(result);
}

void BlockCpu::print() {
	cout << "########################################################################################################################################################################################################" << endl;
	
	cout << endl;
	cout << "BlockCpu from node #" << nodeNumber << endl;
	cout << "Dimension    " << dimension << endl;
	cout << endl;
	cout << "xCount:      " << xCount << endl;
	cout << "yCount:      " << yCount << endl;
	cout << "zCount:      " << zCount << endl;
	cout << endl;
	cout << "xOffset:     " << xOffset << endl;
	cout << "yOffset:     " << yOffset << endl;
	cout << "zOffset:     " << zOffset << endl;
	cout << endl;
	cout << "Cell size:   " << cellSize << endl;
	cout << "Halo size:   " << haloSize << endl;
	
	cout << endl;
	cout << "Block matrix:" << endl;
	cout.setf(ios::fixed);


	// TODO вывод информации о Solver'е

	cout << endl;
	cout << "Send border info (" << countSendSegmentBorder << ")" << endl;
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;
		cout << "Block border #" << i << endl;
		cout << "	Memory address: " << blockBorder[i] << endl;
		cout << "	Memory type:    " << getMemoryTypeName( blockBorderMemoryAllocType[i] ) << endl;
		cout << "	Side:           " << getSideName( sendBorderInfo[index + SIDE] ) << endl;
		cout << "	mOffset:        " << sendBorderInfo[index + M_OFFSET] << endl;
		cout << "	nOffset:        " << sendBorderInfo[index + N_OFFSET] << endl;
		cout << "	mLength:        " << sendBorderInfo[index + M_LENGTH] << endl;
		cout << "	nLength:        " << sendBorderInfo[index + N_LENGTH] << endl;
		cout << endl;
	}

	cout << endl << endl;
	cout << "Receive border info (" << countReceiveSegmentBorder << ")" << endl;
	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;
		cout << "Block border #" << i << endl;
		cout << "	Memory address: " << externalBorder[i] << endl;
		cout << "	Memory type:    " << getMemoryTypeName( externalBorderMemoryAllocType[i] ) << endl;
		cout << "	Side:           " << getSideName( receiveBorderInfo[index + SIDE] ) << endl;
		cout << "	mOffset:        " << receiveBorderInfo[index + M_OFFSET] << endl;
		cout << "	nOffset:        " << receiveBorderInfo[index + N_OFFSET] << endl;
		cout << "	mLength:        " << receiveBorderInfo[index + M_LENGTH] << endl;
		cout << "	nLength:        " << receiveBorderInfo[index + N_LENGTH] << endl;
		cout << endl;
	}

	cout << endl << endl;
	cout << "Parameters (" << mParamsCount << ")" << endl;
	for (int i = 0; i < mParamsCount; ++i) {
		cout << "	parameter #" << i << ":   " << mParams[i] << endl;
	}


	cout << endl << endl;
	cout << "Compute function number" << endl;
	cout.setf(ios::fixed);
	for (int i = 0; i < zCount; ++i) {
		cout << "z = " << i << endl;

		int zShift = xCount * yCount * i;

		for (int j = 0; j < yCount; ++j) {
			int yShift = xCount * j;

			for (int k = 0; k < xCount; ++k) {
				int xShift = k;
				cout << mCompFuncNumber[ zShift + yShift + xShift ] << " ";
			}
			cout << endl;
		}
	}
	cout << endl;

	cout << "########################################################################################################################################################################################################" << endl;
	cout << endl << endl;
}

void BlockCpu::moveTempBorderVectorToBorderArray() {
	blockBorder = new double* [countSendSegmentBorder];
	blockBorderMemoryAllocType = new int [countSendSegmentBorder];
	sendBorderInfo = new int [INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder];

	externalBorder = new double* [countReceiveSegmentBorder];
	externalBorderMemoryAllocType = new int [countReceiveSegmentBorder];
	receiveBorderInfo = new int [INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder];

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);
		blockBorderMemoryAllocType[i] = tempBlockBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		sendBorderInfo[ index + SIDE ] = tempSendBorderInfo.at(index + 0);
		sendBorderInfo[ index + M_OFFSET ] = tempSendBorderInfo.at(index + 1);
		sendBorderInfo[ index + N_OFFSET ] = tempSendBorderInfo.at(index + 2);
		sendBorderInfo[ index + M_LENGTH ] = tempSendBorderInfo.at(index + 3);
		sendBorderInfo[ index + N_LENGTH ] = tempSendBorderInfo.at(index + 4);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);
		externalBorderMemoryAllocType[i] = tempExternalBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		receiveBorderInfo[ index + SIDE ] = tempReceiveBorderInfo.at(index + 0);
		receiveBorderInfo[ index + M_OFFSET ] = tempReceiveBorderInfo.at(index + 1);
		receiveBorderInfo[ index + N_OFFSET ] = tempReceiveBorderInfo.at(index + 2);
		receiveBorderInfo[ index + M_LENGTH ] = tempReceiveBorderInfo.at(index + 3);
		receiveBorderInfo[ index + N_LENGTH ] = tempReceiveBorderInfo.at(index + 4);
	}

	tempBlockBorder.clear();
	tempExternalBorder.clear();
	
	tempBlockBorderMemoryAllocType.clear();
	tempExternalBorderMemoryAllocType.clear();

	tempSendBorderInfo.clear();
	tempReceiveBorderInfo.clear();
}

void BlockCpu::loadData(double* data) {
	cout << endl << "LOAD DATA NOT WORK!" << endl;
	return;
	/*for(int i = 0; i < length * width; i++)
		matrix[i] = data[i];*/
}

void BlockCpu::prepareBorder(int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) {
	double* source = NULL;
	// TODO ПРОВЕРИТЬ!!!! ПОЛУЧАТЬ ОТ СОЛВЕРА!!!
	cout << endl << "source == NULL!!!!" << endl;
	int index = 0;
	for (int z = zStart; z < zStop; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = yStart; y < yStop; ++y) {
			int yShift = xCount * y;

			for (int x = xStart; x < xStop; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}
}

void BlockCpu::createSolver(int solverIdx) {
	/*if      (solverIdx == EULER)
		return new EulerSolver(count);
	else if (solverIdx == RK4)
		return new EulerSolver(count);
	else
		return new EulerSolver(count);*/

	int count = getGridElementCount();

	switch (solverIdx) {
		case EULER:
			mSolver = new EulerSolverCpu(count);
			break;
		case RK4:
			mSolver = new RK4SolverCpu(count);
			break;
		case DP45:
			mSolver = new DP45SolverCpu(count);
			break;
		default:
			mSolver = new EulerSolverCpu(count);
			break;
	}
}

double* BlockCpu::getNewBlockBorder(Block* neighbor, int borderLength, int& memoryType) {
	double* tmpBorder;

	if( ( nodeNumber == neighbor->getNodeNumber() ) && isGPU( neighbor->getBlockType() ) ) {
		cudaMallocHost ( (void**)&tmpBorder, borderLength * sizeof(double) );
		//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
		memoryType = CUDA_MALLOC_HOST;
	}
	else {
		tmpBorder = new double [borderLength];
		//tempBlockBorderMemoryAllocType.push_back(NEW);
		memoryType = NEW;
	}

	return tmpBorder;
}

double* BlockCpu::getNewExternalBorder(Block* neighbor, int borderLength, double* border, int& memoryType) {
	double* tmpBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		tmpBorder = border;
		//tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
		memoryType = NOT_ALLOC;
	}
	else {
		tmpBorder = new double [borderLength];
		//tempExternalBorderMemoryAllocType.push_back(NEW);
		memoryType = NEW;
	}

	return tmpBorder;
}
