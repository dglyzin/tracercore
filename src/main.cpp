#include "BlockCpu.h"
#include "Interconnect.h"

#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "Domain.h"

int main(int argc, char * argv[]) {
	/*
	 * При запуске обязательно следует указывать 2 аргумента.
	 * Пусть к файлу с исходными данными
	 * Путь к файлу, в который будет записан результат.
	 */

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
	 * TODO Дополнительно класс сборщик
	 */

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

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */
	Domain *d = new Domain(world_rank, world_size, argv[1]);
	MPI_Barrier(MPI_COMM_WORLD);

	if(world_rank == 0) {
		printf("\n\n");
		printf("File:         %s\n", argv[1]);
		printf("Node count:   %d\n", d->getCountGridNodes());
		printf("Repeat count: %d\n", d->getRepeatCount());
	}

	/*
	 * Вычисления.
	 */
	// Получить текущее время
	time1 = MPI_Wtime();
	d->count();
	// Получить текущее время
	time2 = MPI_Wtime();

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	if(world_rank == 0) {
		double calcTime = time2 - time1;
		printf("Time:         %f\n", calcTime);
		printf("Speed:        %f\n", (double)(d->getCountGridNodes()) * d->getRepeatCount() / calcTime);
		printf("\n");
	}

	MPI_Barrier(MPI_COMM_WORLD);

	printf("\nThread #%d CPU blocks: %d, GPU blocks: %d\n", world_rank, d->getCountCpuBlocks(), d->getCountGpuBlocks());

	/*
	 * Сбор и вывод результата.
	 * Передается второй аргумент командной строки - путь к файлу, в который будет записан результат.
	 */
	d->print(argv[2]);

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
