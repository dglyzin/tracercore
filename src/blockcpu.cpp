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
	mSolver = GetCpuSolver(_solverIndex, xCount * yCount * zCount * cellSize );


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
	printf("Func array points to %d \n", (long unsigned int) mUserFuncs );

	//mUserFuncs[0](newMatrix, matrix, 0.0, 2, 2, 0, mParams, NULL);
	printf("Func array points to %d \n", (long unsigned int) mUserInitFuncs );
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

	/*
	 * Типы границ блока. Выделение памяти.
	 * По умолчанию границы задаются функциями, то есть нет границ между блоками.
	 */

	/*
	 * TODO Новая реализация всех границ
	 * соседи писать через вектора, как было ранее
	 * для определения "границ" использовать 5 значение: сторона, начальные координаты, конечный (по 2 штуки) - определяют прямоугольную область
	 */
	
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

/*void BlockCpu::computeOneStepBorder(double dX2, double dY2, double dT) {

	 * Теплопроводность



	 * Параллельное вычисление на максимально возможном количестве потоков.
	 * Максимально возможное количесвто потоков получается из-за самой библиотеки omp
	 * Если явно не указывать, какое именно количесвто нитей необходимо создать, то будет создано макстимально возможное на данный момент.

# pragma omp parallel
	{

		 * Для решения задачи теплопроводности нам необходимо знать несколько значений.
		 * Среди них
		 * значение в ячейке выше
		 * значение в ячейке слева
		 * значение в ячейке снизу
		 * значение в ячейке справа
		 * текущее значение в данной ячейке
		 *
		 * остально данные передаются в функцию в качестве параметров.

	double top, left, bottom, right, cur;

# pragma omp for

	 * Проходим по всем ячейкам матрицы.
	 * Для каждой из них будет выполнен перерасчет.

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j) {
			if( i != 0 && i != length - 1 && j != 0 && j != width - 1 )
				continue;
			
			if( i == 0 )
				if( receiveBorderType[TOP][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 100;
					continue;
				}
				else
					top = externalBorder[	receiveBorderType[TOP][j]	][j - externalBorderMove[	receiveBorderType[TOP][j]	]];
			else
				top = matrix[(i - 1) * width + j];


			if( j == 0 )
				if( receiveBorderType[LEFT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					left = externalBorder[	receiveBorderType[LEFT][i]	][i - externalBorderMove[	receiveBorderType[LEFT][i]		]];
			else
				left = matrix[i * width + (j - 1)];


			if( i == length - 1 )
				if( receiveBorderType[BOTTOM][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					bottom = externalBorder[	receiveBorderType[BOTTOM][j]	][j - externalBorderMove[	receiveBorderType[BOTTOM][j]	]];
			else
				bottom = matrix[(i + 1) * width + j];


			if( j == width - 1 )
				if( receiveBorderType[RIGHT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					right = externalBorder[	receiveBorderType[RIGHT][i]	][i - externalBorderMove[	receiveBorderType[RIGHT][i]	]];
			else
				right = matrix[i * width + (j + 1)];


			cur = matrix[i * width + j];
			newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
	}
}*/
/*
void BlockCpu::computeStageCenter() {

	 * Теплопроводность



	 * Параллельное вычисление на максимально возможном количестве потоков.
	 * Максимально возможное количесвто потоков получается из-за самой библиотеки omp
	 * Если явно не указывать, какое именно количесвто нитей необходимо создать, то будет создано макстимально возможное на данный момент.

# pragma omp parallel
	{

		 * Для решения задачи теплопроводности нам необходимо знать несколько значений.
		 * Среди них
		 * значение в ячейке выше
		 * значение в ячейке слева
		 * значение в ячейке снизу
		 * значение в ячейке справа
		 * текущее значение в данной ячейке
		 *
		 * остально данные передаются в функцию в качестве параметров.

	double top, left, bottom, right, cur;

# pragma omp for

	 * Проходим по всем ячейкам матрицы.
	 * Для каждой из них будет выполнен перерасчет.

	for (int i = 1; i < length - 1; ++i)
		for (int j = 1; j < width - 1; ++j) {
			top = matrix[(i - 1) * width + j];
			left = matrix[i * width + (j - 1)];
			bottom = matrix[(i + 1) * width + j];
			right = matrix[i * width + (j + 1)];

			cur = matrix[i * width + j];

			newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
	}
}*/

void BlockCpu::prepareStageData(int stage) {
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;

		switch (sendBorderInfo[index + SIDE]) {
			case LEFT:
				break;
			case RIGHT:
				break;
			case FRONT:
				break;
			case BACK:
				break;
			case TOP:
				break;
			case BOTTOM:
				break;
			default:
				break;
		}
	}
}

double* BlockCpu::getCurrentState(double* result) {
	int count = getGridNodeCount();
	mSolver->copyState(result);

	return result;
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
	/*for (int i = 0; i < zCount; ++i) {
		cout << "z = " << i << endl;

		int zShift = xCount * yCount * i;

		for (int j = 0; j < yCount; ++j) {
			int yShift = xCount * j;

			for (int k = 0; k < xCount; ++k) {
				int xShift = k;

				cout << "(";
				for (int l = 0; l < cellSize; ++l) {
					int cellShift = l;

					cout.width(5);
					cout.precision(1);
					cout << matrix[ (zShift + yShift + xShift)*cellSize + cellShift ] << " ";
				}
				cout << ")";
			}
			cout << endl;
		}
	}*/

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

	/*cout << endl;
	cout << "TopSendBorderType" << endl;
	for( int i =0; i < width; i++ ) {
		cout.width(4);
		cout << sendBorderType[TOP][i] << " ";
	}
	cout << endl;

	cout << endl;
	cout << "LeftSendBorderType" << endl;
	for( int i =0; i < length; i++ ) {
		cout.width(4);
		cout << sendBorderType[LEFT][i] << " ";
	}
	cout << endl;

	cout << endl;
	cout << "BottomSendBorderType" << endl;
	for( int i =0; i < width; i++ ) {
		cout.width(4);
		cout << sendBorderType[BOTTOM][i] << " ";
	}
	cout << endl;

	cout << endl;
	cout << "RightSendBorderType" << endl;
	for( int i =0; i < length; i++ ) {
		cout.width(4);
		cout << sendBorderType[RIGHT][i] << " ";
	}
	cout << endl;

	
	cout << endl << endl;

	
	cout << endl;
	cout << "TopRecieveBorderType" << endl;
	for( int i =0; i < width; i++ ) {
		cout.width(4);
		cout << receiveBorderType[TOP][i] << " ";
	}
	cout << endl;

	cout << endl;
	cout << "LeftRecieveBorderType" << endl;
	for( int i =0; i < length; i++ ) {
		cout.width(4);
		cout << receiveBorderType[LEFT][i] << " ";
	}
	cout << endl;

	cout << endl;
	cout << "BottomRecieveBorderType" << endl;
	for( int i =0; i < width; i++ ) {
		cout.width(4);
		cout << receiveBorderType[BOTTOM][i] << " ";
	}
	cout << endl;

	cout << endl;
	cout << "RightRecieveBorderType" << endl;
	for( int i =0; i < length; i++ ) {
		cout.width(4);
		cout << receiveBorderType[RIGHT][i] << " ";
	}
	cout << endl;

	
	cout << endl << endl;

	
	cout << endl;
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		cout << "BlockBorder #" << i << endl;
		cout << "	Memory address: " << blockBorder[i] << endl;
		cout << "	Border move:    " << blockBorderMove[i] << endl;
		cout << endl;
	}
	
	
	cout << endl;
	
		
	cout << endl;
	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		cout << "ExternalBorder #" << i << endl;
		cout << "	Memory address: " << externalBorder[i] << endl;
		cout << "	Border move:    " << externalBorderMove[i] << endl;
		cout << endl;
	}*/

	cout << "########################################################################################################################################################################################################" << endl;
	cout << endl << endl;
}

double* BlockCpu::addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) {
	countSendSegmentBorder++;

	tempSendBorderInfo.push_back(side);
	tempSendBorderInfo.push_back(mOffset);
	tempSendBorderInfo.push_back(nOffset);
	tempSendBorderInfo.push_back(mLength);
	tempSendBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newBlockBorder;

	if( ( nodeNumber == neighbor->getNodeNumber() ) && isGPU( neighbor->getBlockType() ) ) {
		cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
		tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
	}
	else {
		newBlockBorder = new double [borderLength];
		tempBlockBorderMemoryAllocType.push_back(NEW);
	}

	tempBlockBorder.push_back(newBlockBorder);

	return newBlockBorder;


	/*for (int i = 0; i < borderLength; ++i)
		sendBorderType[side][i + move] = countSendSegmentBorder;

	countSendSegmentBorder++;

	double* newBlockBorder;

	if( ( nodeNumber == neighbor->getNodeNumber() ) && isGPU( neighbor->getBlockType() ) ) {
		cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
		tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
	}
	else {
		newBlockBorder = new double [borderLength];
		tempBlockBorderMemoryAllocType.push_back(NEW);
	}

	tempBlockBorder.push_back(newBlockBorder);
	tempBlockBorderMove.push_back(move);

	return newBlockBorder;
	return NULL;*/
}

double* BlockCpu::addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) {
	countReceiveSegmentBorder++;

	tempReceiveBorderInfo.push_back(side);
	tempReceiveBorderInfo.push_back(mOffset);
	tempReceiveBorderInfo.push_back(nOffset);
	tempReceiveBorderInfo.push_back(mLength);
	tempReceiveBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newExternalBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		newExternalBorder = border;
		tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
	}
	else {
		newExternalBorder = new double [borderLength];
		tempExternalBorderMemoryAllocType.push_back(NEW);
	}

	tempExternalBorder.push_back(newExternalBorder);

	return newExternalBorder;

	/*for (int i = 0; i < borderLength; ++i)
		receiveBorderType[side][i + move] = countReceiveSegmentBorder;

	countReceiveSegmentBorder++;

	double* newExternalBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		newExternalBorder = border;
		tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
	}
	else {
		newExternalBorder = new double [borderLength];
		tempExternalBorderMemoryAllocType.push_back(NEW);
	}

	tempExternalBorder.push_back(newExternalBorder);
	tempExternalBorderMove.push_back(move);

	return newExternalBorder;
	return NULL;*/
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

void BlockCpu::prepareLeftBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength) {
	/*int index = 0;
	for (int z = mOffset; z < mOffset + mLength; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = nOffset; y < nOffset + nLength; ++y) {
			int yShift = xCount * y;

			for (int x = 0; x < haloSize; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}*/

	prepareBorder(source, borderNumber, mOffset, mOffset + mLength, nOffset, nOffset + nLength, 0, haloSize);
}

void BlockCpu::prepareRightBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength) {
	/*int index = 0;
	for (int z = mOffset; z < mOffset + mLength; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = nOffset; y < nOffset + nLength; ++y) {
			int yShift = xCount * y;

			for (int x = xCount - haloSize; x < xCount; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}*/

	prepareBorder(source, borderNumber, mOffset, mOffset + mLength, nOffset, nOffset + mLength, xCount - haloSize, xCount);
}

void BlockCpu::prepareFrontBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength) {
	/*int index = 0;
	for (int z = mOffset; z < mOffset + mLength; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = 0; y < haloSize; ++y) {
			int yShift = xCount * y;

			for (int x = nOffset; x < nOffset + nLength; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}*/

	prepareBorder(source, borderNumber, mOffset, mOffset + nLength, 0, haloSize, nOffset, nOffset + nLength);
}

void BlockCpu::prepareBackBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength) {
	/*int index = 0;
	for (int z = mOffset; z < mOffset + mLength; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = yCount - haloSize; y < yCount; ++y) {
			int yShift = xCount * y;

			for (int x = nOffset; x < nOffset + nLength; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}*/

	prepareBorder(source, borderNumber, mOffset, mOffset + mLength, yCount - haloSize, yCount, nOffset, nOffset + nLength);
}

void BlockCpu::prepareTopBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength) {
	/*int index = 0;
	for (int z = 0; z < haloSize; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = mOffset; y < mOffset + mLength; ++y) {
			int yShift = xCount * y;

			for (int x = nOffset; x < nOffset + nLength; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}*/

	prepareBorder(source, borderNumber, 0, haloSize, mOffset, mOffset + mLength, nOffset, nOffset + nLength);
}

void BlockCpu::prepareBottomBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength) {
	/*int index = 0;
	for (int z = zCount - haloSize; z < zCount; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = mOffset; y < mOffset + mLength; ++y) {
			int yShift = xCount * y;

			for (int x = nOffset; x < nOffset + nLength; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}*/

	prepareBorder(source, borderNumber, zCount - haloSize, zCount, mOffset, mOffset + mLength, nOffset, nOffset + nLength);
}

void BlockCpu::prepareBorder(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) {
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
