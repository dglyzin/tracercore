/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockCpu.h"

using namespace std;

BlockCpu::BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank) : Block(  _length, _width, _lengthMove, _widthMove, _world_rank  ) {

	matrix = new double* [length];
	newMatrix = new double* [length];

	for(int i = 0; i < length; i++) {
		matrix[i] = new double [width];
		newMatrix[i] = new double [width];
	}

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j)
			matrix[i][j] = newMatrix[i][j] = 0;

	/*
	 * Типы границ блока. Выделение памяти.
	 * По умолчанию границы задаются функциями, то есть нет границ между блоками.
	 */
	borderType = new int* [BORDER_COUNT];

	borderType[TOP] = new int[width];
	for(int i = 0; i < width; i++)
		borderType[TOP][i] = BY_FUNCTION;

	borderType[LEFT] = new int[length];
	for (int i = 0; i < length; ++i)
		borderType[LEFT][i] = BY_FUNCTION;

	borderType[BOTTOM] = new int[width];
	for(int i = 0; i < width; i++)
		borderType[BOTTOM][i] = BY_FUNCTION;

	borderType[RIGHT] = new int[length];
	for (int i = 0; i < length; ++i)
		borderType[RIGHT][i] = BY_FUNCTION;

	/*
	 * Границы самого блока.
	 * Это он будет отдавать. Выделение памяти.
	 */
	blockBorder = new double* [BORDER_COUNT];

	blockBorder[TOP] = new double[width];
	for(int i = 0; i < width; i++)
		blockBorder[TOP][i] = 0;

	blockBorder[LEFT] = new double[length];
	for (int i = 0; i < length; ++i)
		blockBorder[LEFT][i] = 0;

	blockBorder[BOTTOM] = new double[width];
	for(int i = 0; i < width; i++)
		blockBorder[BOTTOM][i] = 0;

	blockBorder[RIGHT] = new double[length];
	for (int i = 0; i < length; ++i)
		blockBorder[RIGHT][i] = 0;

	/*
	 * Внешние границы блока.
	 * Сюда будет приходить информация.
	 */
	externalBorder = new double* [BORDER_COUNT];

	externalBorder[TOP] = new double[width];
	for(int i = 0; i < width; i++)
		externalBorder[TOP][i] = 100;//100 * cos( (i - width/2. ) / (width/2.));

	externalBorder[LEFT] = new double[length];
	for (int i = 0; i < length; ++i)
		externalBorder[LEFT][i] = 10;

	externalBorder[BOTTOM] = new double[width];
	for(int i = 0; i < width; i++)
		externalBorder[BOTTOM][i] = 10;

	externalBorder[RIGHT] = new double[length];
	for (int i = 0; i < length; ++i)
		externalBorder[RIGHT][i] = 10;
}

BlockCpu::~BlockCpu() {
	// TODO Auto-generated destructor stub
}

void BlockCpu::courted(double dX2, double dY2, double dT) {
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
	// Проходим по всем ячейкам матрицы.
	// Для каждой из них будет выполнен перерасчет.
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
				if( borderType[TOP][j] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[TOP][j];
					continue;
				}
				else
					top = externalBorder[TOP][j];
			else
				/*
				 * Если находимся не на верхней границе блока, то есть возможность просто получить значение в ячейке выше данной.
				 */
				top = matrix[i - 1][j];


			/*
			 * Аналогично предыдущему случаю.
			 * Только здесь проверка на левую границу блока.
			 *
			 * Рассуждения полностью совпадают со случаем верхней границы.
			 */
			if( j == 0 )
				if( borderType[LEFT][i] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[LEFT][i];
					continue;
				}
				else
					left = externalBorder[LEFT][i];
			else
				left = matrix[i][j - 1];


			/*
			 * Аналогично первому случаю.
			 * Граница нижняя.
			 */
			if( i == length - 1 )
				if( borderType[BOTTOM][j] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[BOTTOM][j];
					continue;
				}
				else
					bottom = externalBorder[BOTTOM][j];
			else
				bottom = matrix[i + 1][j];


			/*
			 * Аналогично первому случаю.
			 * Граница правая.
			 */
			if( j == width - 1 )
				if( borderType[RIGHT][i] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[RIGHT][i];
					continue;
				}
				else
					right = externalBorder[RIGHT][i];
			else
				right = matrix[i][j + 1];


			/*
			 * Текущее значение всегда (если вообще дошли до этого места) можно просто получить из матрицы.
			 */
			cur = matrix[i][j];

			/*
			 * Формула расчета для конкретной точки.
			 */
			newMatrix[i][j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
}
/*
 * Указатель на старую матрицу запоминается
 * Новая матрица становится текущей
 * Память, занимаемая старой матрицей освобождается.
 */
	double** tmp = matrix;

	matrix = newMatrix;

	newMatrix = tmp;
}

void BlockCpu::prepareData() {
	/*
	 * Копирование данных из матрицы в массивы.
	 * В дальнейшем эти массивы (или их части) будет пеесылаться другим блокам.
	 */
	for (int i = 0; i < width; ++i)
		blockBorder[TOP][i] = matrix[0][i];

	for (int i = 0; i < length; ++i)
		blockBorder[LEFT][i] = matrix[i][0];

	for (int i = 0; i < width; ++i)
		blockBorder[BOTTOM][i] = matrix[length-1][i];

	for (int i = 0; i < length; ++i)
		blockBorder[RIGHT][i] = matrix[i][width-1];
}

void BlockCpu::setPartBorder(int type, int side, int move, int borderLength) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	/*
	 * Изначально все границы считаются заданными функциями.
	 * Здесь присваиваются новые значение. Обычно граница с другим блоком.
	 * Хотя фактически это не обязательно.
	 */
	for (int i = 0; i < borderLength; ++i)
		borderType[side][i + move] = type;
}

void BlockCpu::print() {
	printf("FROM NODE #%d", nodeNumber);

	printf("\nLength: %d, Width: %d\n", length, width);
	printf("\nlengthMove: %d, widthMove: %d\n", lenghtMove, widthMove);

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}


	printf("\ntopBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderType[TOP][i]);
	printf("\n");


	printf("\nleftBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderType[LEFT][i]);
	printf("\n");


	printf("\nbottomBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderType[BOTTOM][i]);
	printf("\n");


	printf("\nrightBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderType[RIGHT][i]);
	printf("\n");


	printf("\ntopBlockBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", blockBorder[TOP][i]);
	printf("\n");


	printf("\nleftBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorder[LEFT][i]);
	printf("\n");


	printf("\nbottomBlockBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", blockBorder[BOTTOM][i]);
	printf("\n");


	printf("\nrightBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorder[RIGHT][i]);
	printf("\n");


	printf("\ntopExternalBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", externalBorder[TOP][i]);
	printf("\n");


	printf("\nleftExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorder[LEFT][i]);
	printf("\n");


	printf("\nbottomExternalBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", externalBorder[BOTTOM][i]);
	printf("\n");


	printf("\nrightExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorder[RIGHT][i]);
	printf("\n");


	printf("\n\n\n\n\n\n\n");
}

void BlockCpu::printMatrix() {
	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}
}
