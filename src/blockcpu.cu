/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "blockcpu.h"

using namespace std;

BlockCpu::BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank) : Block(  _length, _width, _lengthMove, _widthMove, _world_rank  ) {

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
}

BlockCpu::~BlockCpu() {
}

void BlockCpu::computeOneStep(double dX2, double dY2, double dT) {
	double t1 = omp_get_wtime();
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
	
	double t2 = omp_get_wtime();
		
	calcTime += (t2 - t1);
}

void BlockCpu::prepareData() {
	double t1 = omp_get_wtime();
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
	
	double t2 = omp_get_wtime();
	
	prepareTime += (t2 - t1);
}

void BlockCpu::print() {
	printf("FROM NODE #%d", nodeNumber);

	printf("\nLength: %d, Width: %d\n", length, width);
	printf("\nlengthMove: %d, widthMove: %d\n", lenghtMove, widthMove);

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i * width + j]);
		printf("\n");
	}

	printf("\ntopSendBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", sendBorderType[TOP][i]);
	printf("\n");

	printf("\nleftSendBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", sendBorderType[LEFT][i]);
	printf("\n");

	printf("\nbottomSendBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", sendBorderType[BOTTOM][i]);
	printf("\n");

	printf("\nrightSendBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", sendBorderType[RIGHT][i]);
	printf("\n\n\n");



	printf("\ntopRecieveBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", receiveBorderType[TOP][i]);
	printf("\n");

	printf("\nleftRecieveBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", receiveBorderType[LEFT][i]);
	printf("\n");

	printf("\nbottomRecieveBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", receiveBorderType[BOTTOM][i]);
	printf("\n");

	printf("\nrightRecieveBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", receiveBorderType[RIGHT][i]);
	printf("\n");


	for (int i = 0; i < countSendSegmentBorder; ++i)
		printf("\nblockBorder #%d : %d : %d\n", i, blockBorder[i], blockBorderMove[i]);

	for (int i = 0; i < countReceiveSegmentBorder; ++i)
		printf("\nexternalBorder #%d : %d : %d\n", i, externalBorder[i], externalBorderMove[i]);


	printf("\n\n\n");
}

double* BlockCpu::addNewBlockBorder(int nodeNeighbor, int typeNeighbor, int side, int move, int borderLength) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	for (int i = 0; i < borderLength; ++i)
		sendBorderType[side][i + move] = countSendSegmentBorder;

	countSendSegmentBorder++;

	double* newBlockBorder;

	if( (nodeNumber == nodeNeighbor) && isGPU(typeNeighbor) )
		cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
	else
		newBlockBorder = new double [borderLength];

	tempBlockBorder.push_back(newBlockBorder);
	tempBlockBorderMove.push_back(move);

	return newBlockBorder;
}

double* BlockCpu::addNewExternalBorder(int nodeNeighbor, int side, int move, int borderLength, double* border) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	for (int i = 0; i < borderLength; ++i)
		receiveBorderType[side][i + move] = countReceiveSegmentBorder;

	countReceiveSegmentBorder++;

	double* newExternalBorder;

	if( nodeNumber == nodeNeighbor )
		newExternalBorder = border;
	else
		newExternalBorder = new double [borderLength];

	tempExternalBorder.push_back(newExternalBorder);
	tempExternalBorderMove.push_back(move);

	return newExternalBorder;
}

void BlockCpu::moveTempBorderVectorToBorderArray() {
	blockBorder = new double* [countSendSegmentBorder];
	blockBorderMove = new int [countSendSegmentBorder];

	externalBorder = new double* [countReceiveSegmentBorder];
	externalBorderMove = new int [countReceiveSegmentBorder];

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);
		blockBorderMove[i] = tempBlockBorderMove.at(i);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);
		externalBorderMove[i] = tempExternalBorderMove.at(i);
	}

	tempBlockBorder.clear();
	tempBlockBorderMove.clear();
	tempExternalBorder.clear();
	tempExternalBorderMove.clear();
}
