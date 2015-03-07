/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "blockcpu.h"

using namespace std;

BlockCpu::BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber) : Block(  _length, _width, _lengthMove, _widthMove, _nodeNumber, _deviceNumber  ) {

	matrix = new double [length * width];
	newMatrix = new double [length * width];

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j)
			matrix[i * width + j] = newMatrix[i * width + j] = 0;

	/*
	 * Типы границ блока. Выделение памяти.
	 * По умолчанию границы задаются функциями, то есть нет границ между блоками.
	 */
	sendBorderType = new int* [BORDER_COUNT];

	sendBorderType[TOP] = new int[width];
	for(int i = 0; i < width; i++)
		sendBorderType[TOP][i] = BY_FUNCTION;

	sendBorderType[LEFT] = new int[length];
	for (int i = 0; i < length; ++i)
		sendBorderType[LEFT][i] = BY_FUNCTION;

	sendBorderType[BOTTOM] = new int[width];
	for(int i = 0; i < width; i++)
		sendBorderType[BOTTOM][i] = BY_FUNCTION;

	sendBorderType[RIGHT] = new int[length];
	for (int i = 0; i < length; ++i)
		sendBorderType[RIGHT][i] = BY_FUNCTION;


	receiveBorderType = new int* [BORDER_COUNT];

	receiveBorderType[TOP] = new int[width];
	for(int i = 0; i < width; i++)
		receiveBorderType[TOP][i] = BY_FUNCTION;

	receiveBorderType[LEFT] = new int[length];
	for (int i = 0; i < length; ++i)
		receiveBorderType[LEFT][i] = BY_FUNCTION;

	receiveBorderType[BOTTOM] = new int[width];
	for(int i = 0; i < width; i++)
		receiveBorderType[BOTTOM][i] = BY_FUNCTION;

	receiveBorderType[RIGHT] = new int[length];
	for (int i = 0; i < length; ++i)
		receiveBorderType[RIGHT][i] = BY_FUNCTION;
	
	
	result = new double [length * width];
}

BlockCpu::~BlockCpu() {
	if(matrix != NULL)
		delete matrix;
	
	if(newMatrix != NULL)
		delete newMatrix;
	
	if(sendBorderType != NULL) {
		if(sendBorderType[TOP] != NULL)
			delete sendBorderType[TOP];
		
		if(sendBorderType[LEFT] != NULL)
			delete sendBorderType[LEFT];
		
		if(sendBorderType[BOTTOM] != NULL)
			delete sendBorderType[BOTTOM];
		
		if(sendBorderType[RIGHT] != NULL)
			delete sendBorderType[RIGHT];
		
		delete sendBorderType;		
	}
	
	if(receiveBorderType != NULL) {
		if(receiveBorderType[TOP] != NULL)
			delete receiveBorderType[TOP];
		
		if(receiveBorderType[LEFT] != NULL)
			delete receiveBorderType[LEFT];
		
		if(receiveBorderType[BOTTOM] != NULL)
			delete receiveBorderType[BOTTOM];
		
		if(receiveBorderType[RIGHT] != NULL)
			delete receiveBorderType[RIGHT];
		
		delete receiveBorderType;		
	}
	
	
	if(blockBorder != NULL) {
		for(int i = 0; i < countSendSegmentBorder; i++ )
			freeMemory(blockBorderMemoryAllocType[i], blockBorder[i]);
		
		delete blockBorder;
		delete blockBorderMemoryAllocType;
	}
	
	if(blockBorderMove != NULL)
		delete blockBorderMove;
	
	
	if(externalBorder != NULL) {
		for(int i = 0; i < countReceiveSegmentBorder; i++ )
			freeMemory(externalBorderMemoryAllocType[i], externalBorder[i]);
		
		delete externalBorder;
		delete externalBorderMemoryAllocType;
	}
	
	if(externalBorderMove != NULL)
		delete externalBorderMove;
	
	if(result != NULL)
		delete result;
}

void BlockCpu::computeOneStep(double dX2, double dY2, double dT) {
	/*
	 * Теплопроводность
	 */

	/*
	 * Параллельное вычисление на максимально возможном количестве потоков.
	 * Максимально возможное количесвто потоков получается из-за самой библиотеки omp
	 * Если явно не указывать, какое именно количесвто нитей необходимо создать, то будет создано макстимально возможное на данный момент.
	 */
# pragma omp parallel
	{
		/*
		 * Для решения задачи теплопроводности нам необходимо знать несколько значений.
		 * Среди них
		 * значение в ячейке выше
		 * значение в ячейке слева
		 * значение в ячейке снизу
		 * значение в ячейке справа
		 * текущее значение в данной ячейке
		 *
		 * остально данные передаются в функцию в качестве параметров.
		 */
	double top, left, bottom, right, cur;

# pragma omp for
	/*
	 * Проходим по всем ячейкам матрицы.
	 * Для каждой из них будет выполнен перерасчет.
	 */
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j) {
			/*
			 * Если находимся на верхней границе блока.
			 * В таком случае необходимо проверить тип границы и в зависимости от ответа принать решение.
			 *
			 * Стоит отличать границу реальную от границы с блоком.
			 * Если граница реальна, то точка на границе может не иметь значения выше / значения ниже и так далее, так как это реально границе ВСЕЙ ОБЛАСТИ.
			 * Если эта граница с другим блоком, то значение выше / ниже сущесвтуют, так как это не граница области.
			 * Значит их нужно получить и использовать при ирасчете нового значения.
			 */
			
			if( i == 0 )
				/*
				 * На данный момент есть только 2 типа границы. Функция и другой блок.
				 * Поэтому использование else корректно.
				 *
				 * Если граница задана функцией, то это значит,
				 * что наданном этапе в массиве externalBorder уже должны лежать свежие данные от функции.
				 * В таком случае просто копируем данные из массива в матрицу. Для этой ячейки расчет окончен.
				 *
				 * Если это граница с другим блоком, то в top (значение в ячейке выше данной) записываем информацию с гранцы.
				 * Но продолжаем расчет.
				 */
				if( receiveBorderType[TOP][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 100;
					continue;
				}
				else
					top = externalBorder[	receiveBorderType[TOP][j]	][j - externalBorderMove[	receiveBorderType[TOP][j]	]];
			else
				/*
				 * Если находимся не на верхней границе блока, то есть возможность просто получить значение в ячейке выше данной.
				 */
				top = matrix[(i - 1) * width + j];


			/*
			 * Аналогично предыдущему случаю.
			 * Только здесь проверка на левую границу блока.
			 *
			 * Рассуждения полностью совпадают со случаем верхней границы.
			 */
			if( j == 0 )
				if( receiveBorderType[LEFT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					left = externalBorder[	receiveBorderType[LEFT][i]	][i - externalBorderMove[	receiveBorderType[LEFT][i]		]];
			else
				left = matrix[i * width + (j - 1)];


			/*
			 * Аналогично первому случаю.
			 * Граница нижняя.
			 */
			if( i == length - 1 )
				if( receiveBorderType[BOTTOM][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					bottom = externalBorder[	receiveBorderType[BOTTOM][j]	][j - externalBorderMove[	receiveBorderType[BOTTOM][j]	]];
			else
				bottom = matrix[(i + 1) * width + j];


			/*
			 * Аналогично первому случаю.
			 * Граница правая.
			 */
			if( j == width - 1 )
				if( receiveBorderType[RIGHT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					right = externalBorder[	receiveBorderType[RIGHT][i]	][i - externalBorderMove[	receiveBorderType[RIGHT][i]	]];
			else
				right = matrix[i * width + (j + 1)];


			/*
			 * Текущее значение всегда (если вообще дошли до этого места) можно просто получить из матрицы.
			 */
			cur = matrix[i * width + j];

			/*
			 * Формула расчета для конкретной точки.
			 */
			newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
	}
/*
 * Указатель на старую матрицу запоминается
 * Новая матрица становится текущей
 * Память, занимаемая старой матрицей освобождается.
 */
	double* tmp = matrix;

	matrix = newMatrix;

	newMatrix = tmp;
}

void BlockCpu::computeOneStepBorder(double dX2, double dY2, double dT) {
	/*
	 * Теплопроводность
	 */

	/*
	 * Параллельное вычисление на максимально возможном количестве потоков.
	 * Максимально возможное количесвто потоков получается из-за самой библиотеки omp
	 * Если явно не указывать, какое именно количесвто нитей необходимо создать, то будет создано макстимально возможное на данный момент.
	 */
# pragma omp parallel
	{
		/*
		 * Для решения задачи теплопроводности нам необходимо знать несколько значений.
		 * Среди них
		 * значение в ячейке выше
		 * значение в ячейке слева
		 * значение в ячейке снизу
		 * значение в ячейке справа
		 * текущее значение в данной ячейке
		 *
		 * остально данные передаются в функцию в качестве параметров.
		 */
	double top, left, bottom, right, cur;

# pragma omp for
	/*
	 * Проходим по всем ячейкам матрицы.
	 * Для каждой из них будет выполнен перерасчет.
	 */
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j) {
			/*
			 * Если находимся на верхней границе блока.
			 * В таком случае необходимо проверить тип границы и в зависимости от ответа принать решение.
			 *
			 * Стоит отличать границу реальную от границы с блоком.
			 * Если граница реальна, то точка на границе может не иметь значения выше / значения ниже и так далее, так как это реально границе ВСЕЙ ОБЛАСТИ.
			 * Если эта граница с другим блоком, то значение выше / ниже сущесвтуют, так как это не граница области.
			 * Значит их нужно получить и использовать при ирасчете нового значения.
			 */
			
			if( i == 0 )
				/*
				 * На данный момент есть только 2 типа границы. Функция и другой блок.
				 * Поэтому использование else корректно.
				 *
				 * Если граница задана функцией, то это значит,
				 * что наданном этапе в массиве externalBorder уже должны лежать свежие данные от функции.
				 * В таком случае просто копируем данные из массива в матрицу. Для этой ячейки расчет окончен.
				 *
				 * Если это граница с другим блоком, то в top (значение в ячейке выше данной) записываем информацию с гранцы.
				 * Но продолжаем расчет.
				 */
				if( receiveBorderType[TOP][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 100;
					continue;
				}
				else
					top = externalBorder[	receiveBorderType[TOP][j]	][j - externalBorderMove[	receiveBorderType[TOP][j]	]];


			/*
			 * Аналогично предыдущему случаю.
			 * Только здесь проверка на левую границу блока.
			 *
			 * Рассуждения полностью совпадают со случаем верхней границы.
			 */
			if( j == 0 )
				if( receiveBorderType[LEFT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					left = externalBorder[	receiveBorderType[LEFT][i]	][i - externalBorderMove[	receiveBorderType[LEFT][i]		]];


			/*
			 * Аналогично первому случаю.
			 * Граница нижняя.
			 */
			if( i == length - 1 )
				if( receiveBorderType[BOTTOM][j] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					bottom = externalBorder[	receiveBorderType[BOTTOM][j]	][j - externalBorderMove[	receiveBorderType[BOTTOM][j]	]];


			/*
			 * Аналогично первому случаю.
			 * Граница правая.
			 */
			if( j == width - 1 )
				if( receiveBorderType[RIGHT][i] == BY_FUNCTION ) {
					newMatrix[i * width + j] = 10;
					continue;
				}
				else
					right = externalBorder[	receiveBorderType[RIGHT][i]	][i - externalBorderMove[	receiveBorderType[RIGHT][i]	]];

			cur = matrix[i * width + j];

			newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
	}
}

void BlockCpu::computeOneStepCenter(double dX2, double dY2, double dT) {
	/*
	 * Теплопроводность
	 */

	/*
	 * Параллельное вычисление на максимально возможном количестве потоков.
	 * Максимально возможное количесвто потоков получается из-за самой библиотеки omp
	 * Если явно не указывать, какое именно количесвто нитей необходимо создать, то будет создано макстимально возможное на данный момент.
	 */
# pragma omp parallel
	{
		/*
		 * Для решения задачи теплопроводности нам необходимо знать несколько значений.
		 * Среди них
		 * значение в ячейке выше
		 * значение в ячейке слева
		 * значение в ячейке снизу
		 * значение в ячейке справа
		 * текущее значение в данной ячейке
		 *
		 * остально данные передаются в функцию в качестве параметров.
		 */
	double top, left, bottom, right, cur;

# pragma omp for
	/*
	 * Проходим по всем ячейкам матрицы.
	 * Для каждой из них будет выполнен перерасчет.
	 */
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
}

void BlockCpu::prepareData() {
	/*
	 * Копирование данных из матрицы в массивы.
	 * В дальнейшем эти массивы будет пеесылаться другим блокам.
	 */
	for (int i = 0; i < width; ++i)
		if( sendBorderType[TOP][i] != BY_FUNCTION )
			blockBorder[	sendBorderType[TOP][i]	][i - blockBorderMove[	sendBorderType[TOP][i]	]] = matrix[0 * width + i];

	for (int i = 0; i < length; ++i)
		if( sendBorderType[LEFT][i] != BY_FUNCTION )
			blockBorder[	sendBorderType[LEFT][i]	][i - blockBorderMove[	sendBorderType[LEFT][i]	]] = matrix[i * width + 0];

	for (int i = 0; i < width; ++i)
		if( sendBorderType[BOTTOM][i] != BY_FUNCTION )
			blockBorder[	sendBorderType[BOTTOM][i]	][i - blockBorderMove[	sendBorderType[BOTTOM][i]	]] = matrix[(length - 1) * width + i];

	for (int i = 0; i < length; ++i)
		if( sendBorderType[RIGHT][i] != BY_FUNCTION )
			blockBorder[	sendBorderType[RIGHT][i]	][i - blockBorderMove[	sendBorderType[RIGHT][i]	]] = matrix[i * width + (width - 1)];
}

double* BlockCpu::getResult() {
	for(int i = 0; i < length * width; i++)
		result[i] = matrix[i];
	
	return result;
}

void BlockCpu::print() {
	cout << "########################################################################################################################################################################################################" << endl;
	
	cout << endl;
	cout << "BlockCpu from node #" << nodeNumber << endl;
	cout << "Length:      " << length << endl;
	cout << "Width :      " << width << endl;
	cout << endl;
	cout << "Length move: " << lengthMove << endl;
	cout << "Width move:  " << widthMove << endl;
	
	cout << endl;
	cout << "Block matrix:" << endl;
	cout.setf(ios::fixed);
	for(int i = 0; i < length; i++) {
		for( int j = 0; j < width; j++ ) {
			cout.width(7);
			cout.precision(1);
			cout << matrix[i * width + j];
		}
		cout << endl;
	}
	
	cout << endl;
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
	}

	cout << "########################################################################################################################################################################################################" << endl;
	cout << endl << endl;
}

double* BlockCpu::addNewBlockBorder(Block* neighbor, int side, int move, int borderLength) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	for (int i = 0; i < borderLength; ++i)
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
}

double* BlockCpu::addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	for (int i = 0; i < borderLength; ++i)
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
}

void BlockCpu::moveTempBorderVectorToBorderArray() {
	blockBorder = new double* [countSendSegmentBorder];
	blockBorderMove = new int [countSendSegmentBorder];
	blockBorderMemoryAllocType = new int [countSendSegmentBorder];

	externalBorder = new double* [countReceiveSegmentBorder];
	externalBorderMove = new int [countReceiveSegmentBorder];
	externalBorderMemoryAllocType = new int [countReceiveSegmentBorder];	
	

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);
		blockBorderMove[i] = tempBlockBorderMove.at(i);
		blockBorderMemoryAllocType[i] = tempBlockBorderMemoryAllocType.at(i);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);
		externalBorderMove[i] = tempExternalBorderMove.at(i);
		externalBorderMemoryAllocType[i] = tempExternalBorderMemoryAllocType.at(i);
	}

	tempBlockBorder.clear();
	tempBlockBorderMove.clear();
	tempExternalBorder.clear();
	tempExternalBorderMove.clear();
	
	tempBlockBorderMemoryAllocType.clear();
	tempExternalBorderMemoryAllocType.clear();
}

void BlockCpu::loadData(double* data) {
	for(int i = 0; i < length * width; i++)
		matrix[i] = data[i];
}
