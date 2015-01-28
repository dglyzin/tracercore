#include "BlockCpu.h"
#include "Interconnect.h"

#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "Domain.h"

int main(int argc, char * argv[]) {
	/*
	 * TODO Функции расчета для блоков
	 * TODO Функции расчета в домене для вычисления граничных уловий
	 * TODO Пересылка между видеокартами и центральными процессора на разных потоках исполения
	 * TODO Реализовать работу с помощью видеокарт. Класс BlockGpu.
	 *
	 * TODO Реализовать чтение с файла.
	 * TODO [Done!] Потоки должны знать кто и кого должен был создать.
	 *
	 * TODO [Done!] Блок должен знать свои координаты!!!
	 * TODO Выделять память на границы блоков после создания соединений
	 */
	MPI_Init(NULL, NULL);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	double time1, time2;


	/*
	 * Количество итераций вычисления.
	 * В будущем должна стать вычисляемой вличиной.
	 */

	/*
	 * Создание основного управляющего класса.
	 */
	Domain *d = new Domain(world_rank, world_size, argv[1]);

	/*
	 * Вычисления.
	 */
	time1 = MPI_Wtime();
	d->count();
	time2 = MPI_Wtime();

	if(world_rank == 0)
		printf("\n\nCount grid nodes: %d, time: %f\n\n\n", d->getCountGridNodes(), time2 - time1);

	/*
	 * Сбор и вывод результата.
	 */
	d->print(argv[2]);

	MPI_Finalize();
}
