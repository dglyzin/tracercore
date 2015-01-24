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
	 * TODO Потоки должны знать кто и кого должен был создать.
	 *
	 * TODO Блок должен знать свои координаты!!!
	 * TODO Выделять память на границы блоков после создания соединений
	 */
	MPI_Init(NULL, NULL);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


	/*
	 * Количество итераций вычисления.
	 * В будущем должна стать вычисляемой вличиной.
	 */
	int repeatCount = atoi(argv[1]);

	/*
	 * Создание основного управляющего класса.
	 */
	Domain *d = new Domain(world_rank, world_size, 4, 3);

	/*
	 * Вычисления.
	 */
	for (int i = 0; i < repeatCount; ++i)
		d->calc(world_rank, 4, 3);

	/*
	 * Сбор и вывод результата.
	 */
	d->print(world_rank, 4);

	MPI_Finalize();
}
