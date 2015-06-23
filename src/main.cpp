#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "domain.h"

using namespace std;

/*
class A {
public:
	virtual void Am1() = 0;
	virtual void Am2() = 0;
	virtual void Am3() = 0;
};

class B {
public:
	virtual void Bm1() = 0;
	virtual void Bm2() = 0;
	virtual void Bm3() = 0;
};

class C : public A, B {
public:
	void Am1() { return; }

	void Bm1() { return; }
	void Bm2() { return; }
	void Bm3() { return; }

	static int get3() { return 3; }
};

class D : public C {
public:
	void Am2() { return; }
	void Am3() { return; }
};*/


int main(int argc, char * argv[]) {
	/*
	 * Инициализация MPI
	 */

	MPI_Init(NULL, NULL);

	/*
	 * Для каждого потока получаем его номер и общее количество потоков.
	 */
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	/*
	 * Переменные для расчета время работы.
	 */
	double time1, time2;

	char* inputFile = argv[1];

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */
	printf ("DEBUG creating domain...\n ");
	Domain* d = new Domain(world_rank, world_size, inputFile, 0, 0, 0, NULL);

	/*
	 * Вычисления.
	 */
	//omp_set_num_threads(1);

	printf ("Running computations \n");
	time1 = MPI_Wtime();
	d->compute(inputFile);
	time2 = MPI_Wtime();

	if(d->isNan())
		printf("\n\n\n\nNAN!!!\n\n\n\n");

	d->saveState(inputFile);

	//d->printBlocksToConsole();

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	d->printStatisticsInfo(inputFile, NULL, time2 - time1, NULL);

	delete d;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
