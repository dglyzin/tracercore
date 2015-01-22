#include "BlockCpu.h"
#include "Interconnect.h"

#include <mpi.h>
#include <stdlib.h>

int main(int argc, char * argv[]) {

	// Initialize the MPI environment
	  MPI_Init(NULL, NULL);

	  // Get the number of processes
	  int world_size;
	  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	  // Get the rank of the process
	  int world_rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	  int b0_length = 50;
	  int b0_width = 25;

	  int b1_length = 25;
	  int b1_width = 50;

	  int b2_length = 50;
	  int b2_width = 25;

	  int b3_length = 50;
	  int b3_width = 25;

	  int repeatCount = atoi(argv[1]);

	  if(world_rank == 0) {
		  Block* b0 = new BlockCpu(b0_length, b0_width);

		  MPI_Status status;
		  //printf("\n\n******0-0\n\n");

		  int* topBoundaryType = b0->getTopBoundaryType();
		  int* leftBoundaryType = b0->getLeftBoundaryType();
		  int* bottomBoundaryType = b0->getBottomBoundaryType();
		  int* rightBoundaryType = b0->getRightBoundaryType();

		  for (int i = 0; i < b0_width; ++i)
			  topBoundaryType[i] = 1;

		  for (int i = 0; i < b0_length; ++i)
			  leftBoundaryType[i] = 1;

		  for (int i = 0; i < b0_width; ++i)
			  bottomBoundaryType[i] = 1;

		  for (int i = 0; i < 15; ++i)
			  rightBoundaryType[i] = 1;
		  for (int i = 15; i < b1_length+15; ++i)
			  rightBoundaryType[i] = 0;
		  for (int i = b1_length+15; i < b0_length; ++i)
			  rightBoundaryType[i] = 1;


		  double* topBlockBoundary = b0->getTopBlockBoundary();
		  double* leftBlockBoundary = b0->getLeftBlockBoundary();
		  double* bottomBlockBoundary = b0->getBottomBlockBoundary();
		  double* rightBlockBoundary = b0->getRightBlockBoundary();

		  double* topExternalBoundary = b0->getTopExternalBoundary();
		  double* leftExternalBoundary = b0->getLeftExternalBoundary();
		  double* bottomExternalBoundary = b0->getBottomExternalBoundary();
		  double* rightExternalBoundary = b0->getRightExternalBoundary();

		  //printf("\n\n******0-1\n\n");

		  for (int i = 0; i < b0_width; ++i) {
			  topBlockBoundary[i] = 0;
			  bottomBlockBoundary[i] = 0;

			  topExternalBoundary[i] = 100;
			  bottomExternalBoundary[i] = 10;
		  }

		  for (int i = 0; i < b0_length; ++i) {
			  leftBlockBoundary[i] = 0;
			  rightBlockBoundary[i] = 0;

			  leftExternalBoundary[i] = 10;
			  rightExternalBoundary[i] = 10;
		  }

		  //printf("\n\n******0-2\n\n");

		  Interconnect* i2 = new Interconnect(0, 1, 0, 0, b1_length, rightBlockBoundary + 15, NULL);
		  Interconnect* i3 = new Interconnect(1, 0, 0, 0, b1_length, NULL, rightExternalBoundary + 15);

		  //printf("\n\n******0-3\n\n");

		  for (int i = 0; i < repeatCount; ++i) {
			b0->courted();
			b0->prepareData();

			//printf("\n\n******0-4\n\n");

			//b0->print(world_rank);

			MPI_Barrier( MPI_COMM_WORLD );

			//printf("\n\nbefore #0 %d\n\n", i);
			i2->sendRecv(world_rank);
			i3->sendRecv(world_rank);
			//printf("\n\nafter #0 %d\n\n", i);

			MPI_Barrier( MPI_COMM_WORLD );
		  }

		  double** resault = b0->getResault();

		  double** resaultAll = new double* [85];
		  for (int i = 0; i < 85; ++i)
			  resaultAll[i] = new double[100];

		  for (int i = 0; i < 85; ++i)
				for (int j = 0; j < 100; ++j)
					resaultAll[i][j] = 0;

		  for (int i = 0; i < b0_length; ++i)
		  		for (int j = 0; j < b0_width; ++j)
		  			resaultAll[i + 35][j] = resault[i][j];

		  for (int i = 0; i < b1_length; ++i)
			  MPI_Recv(resaultAll[i + 50] + 25, b1_width, MPI_DOUBLE, 1, 999, MPI_COMM_WORLD, &status);

		  for (int i = 0; i < b2_length; ++i)
			  MPI_Recv(resaultAll[i] + 40, b2_width, MPI_DOUBLE, 2, 999, MPI_COMM_WORLD, &status);

		  for (int i = 0; i < b2_length; ++i)
			  MPI_Recv(resaultAll[i + 35] + 75, b3_width, MPI_DOUBLE, 3, 999, MPI_COMM_WORLD, &status);

		  FILE* out = fopen("res", "wb");

		  for (int i = 0; i < 85; ++i) {
			for (int j = 0; j < 100; ++j)
				fprintf(out, "%d %d %f\n", i, j, resaultAll[i][j]);
			fprintf(out, "\n");
		  }

		  fclose(out);
	  }

	  if(world_rank == 1) {
		  Block* b1 = new BlockCpu(b1_length, b1_width);

		  int* topBoundaryType = b1->getTopBoundaryType();
		  int* leftBoundaryType = b1->getLeftBoundaryType();
		  int* bottomBoundaryType = b1->getBottomBoundaryType();
		  int* rightBoundaryType = b1->getRightBoundaryType();


		  for (int i = 0; i < 15; ++i)
			  topBoundaryType[i] = 1;
		  for (int i = 15; i < b2_width + 15; ++i)
			  topBoundaryType[i] = 0;
		  for (int i = b2_width + 15; i < b1_width; ++i)
			  topBoundaryType[i] = 1;

		  for (int i = 0; i < b1_length; ++i)
			  leftBoundaryType[i] = 0;

		  for (int i = 0; i < b1_width; ++i)
			  bottomBoundaryType[i] = 1;

		  for (int i = 0; i < b1_length; ++i)
			  rightBoundaryType[i] = 0;


		  double* topBlockBoundary = b1->getTopBlockBoundary();
		  double* leftBlockBoundary = b1->getLeftBlockBoundary();
		  double* bottomBlockBoundary = b1->getBottomBlockBoundary();
		  double* rightBlockBoundary = b1->getRightBlockBoundary();

		  double* topExternalBoundary = b1->getTopExternalBoundary();
		  double* leftExternalBoundary = b1->getLeftExternalBoundary();
		  double* bottomExternalBoundary = b1->getBottomExternalBoundary();
		  double* rightExternalBoundary = b1->getRightExternalBoundary();

		  //printf("\n\n******1-1\n\n");

		  for (int i = 0; i < b1_width; ++i) {
			  topBlockBoundary[i] = 0;
			  bottomBlockBoundary[i] = 0;

			  topExternalBoundary[i] = 100;
			  bottomExternalBoundary[i] = 10;
		  }

		  for (int i = 0; i < b1_length; ++i) {
			  leftBlockBoundary[i] = 0;
			  rightBlockBoundary[i] = 0;

			  leftExternalBoundary[i] = 10;
			  rightExternalBoundary[i] = 10;
		  }

		  //printf("\n\n******1-2\n\n");


		  Interconnect* i0 = new Interconnect(1, 2, 0, 0, b2_width, topBlockBoundary + 15, NULL);
		  Interconnect* i1 = new Interconnect(2, 1, 0, 0, b2_width, NULL, topExternalBoundary + 15);
		  Interconnect* i2 = new Interconnect(0, 1, 0, 0, b1_length, NULL, leftExternalBoundary);
		  Interconnect* i3 = new Interconnect(1, 0, 0, 0, b1_length, leftBlockBoundary, NULL);
		  Interconnect* i4 = new Interconnect(1, 3, 0, 0, b1_length, rightBlockBoundary, NULL);
		  Interconnect* i5 = new Interconnect(3, 1, 0, 0, b1_length, NULL, rightExternalBoundary);

		  //printf("\n\n******1-3\n\n");

		  for (int i = 0; i < repeatCount; ++i) {
			b1->courted();
			b1->prepareData();

			//printf("\n\n******1-4\n\n");

			MPI_Barrier( MPI_COMM_WORLD );

			//printf("\n\nbefore #1 %d\n\n", i);
			i0->sendRecv(world_rank);
			i1->sendRecv(world_rank);
			i2->sendRecv(world_rank);
			i3->sendRecv(world_rank);
			//printf("\n\nbetween #1\n\n");
			i4->sendRecv(world_rank);
			i5->sendRecv(world_rank);
			//printf("\n\nafter #1 %d\n\n", i);

			MPI_Barrier( MPI_COMM_WORLD );
		  }

		  double** resault = b1->getResault();

		  for (int i = 0; i < b1_length; ++i)
			  MPI_Send(resault[i], b1_width, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
	  }

	  if(world_rank == 2) {
		  Block* b2 = new BlockCpu(b2_length, b2_width);

		  int* topBoundaryType = b2->getTopBoundaryType();
		  int* leftBoundaryType = b2->getLeftBoundaryType();
		  int* bottomBoundaryType = b2->getBottomBoundaryType();
		  int* rightBoundaryType = b2->getRightBoundaryType();

		  //printf("\n\n******2-0\n\n");
		  for (int i = 0; i < b2_width; ++i)
			  topBoundaryType[i] = 1;

		  for (int i = 0; i < b2_length; ++i)
			  leftBoundaryType[i] = 1;

		  for (int i = 0; i < b2_width; ++i)
			  bottomBoundaryType[i] = 0;

		  for (int i = 0; i < b2_length; ++i)
			  rightBoundaryType[i] = 1;


		  double* topBlockBoundary = b2->getTopBlockBoundary();
		  double* leftBlockBoundary = b2->getLeftBlockBoundary();
		  double* bottomBlockBoundary = b2->getBottomBlockBoundary();
		  double* rightBlockBoundary = b2->getRightBlockBoundary();

		  double* topExternalBoundary = b2->getTopExternalBoundary();
		  double* leftExternalBoundary = b2->getLeftExternalBoundary();
		  double* bottomExternalBoundary = b2->getBottomExternalBoundary();
		  double* rightExternalBoundary = b2->getRightExternalBoundary();

		  //printf("\n\n******2-1\n\n");

		  for (int i = 0; i < b2_width; ++i) {
			  topBlockBoundary[i] = 0;
			  bottomBlockBoundary[i] = 0;

			  topExternalBoundary[i] = 10;
			  bottomExternalBoundary[i] = 100;
		  }

		  for (int i = 0; i < b2_length; ++i) {
			  leftBlockBoundary[i] = 0;
			  rightBlockBoundary[i] = 0;

			  leftExternalBoundary[i] = 10;
			  rightExternalBoundary[i] = 10;
		  }

		  //printf("\n\n******2-2\n\n");

		  Interconnect* i0 = new Interconnect(1, 2, 0, 0, b2_width, NULL, bottomExternalBoundary);
		  Interconnect* i1 = new Interconnect(2, 1, 0, 0, b2_width, bottomBlockBoundary, NULL);

		  //printf("\n\n******1-3\n\n");

		  for (int i = 0; i < repeatCount; ++i) {
			b2->courted();
			b2->prepareData();

			//printf("\n\n******2-4\n\n");

			MPI_Barrier( MPI_COMM_WORLD );

			//printf("\n\nbefore #2 %d\n\n", i);
			i0->sendRecv(world_rank);
			i1->sendRecv(world_rank);
			//printf("\n\nafter #2 %d\n\n", i);

			MPI_Barrier( MPI_COMM_WORLD );
		  }

		  double** resault = b2->getResault();

		  for (int i = 0; i < b2_length; ++i)
			  MPI_Send(resault[i], b2_width, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
	  }

	  if(world_rank == 3) {
		  Block* b3 = new BlockCpu(b3_length, b3_width);

		  int* topBoundaryType = b3->getTopBoundaryType();
		  int* leftBoundaryType = b3->getLeftBoundaryType();
		  int* bottomBoundaryType = b3->getBottomBoundaryType();
		  int* rightBoundaryType = b3->getRightBoundaryType();

		  for (int i = 0; i < b3_width; ++i)
			  topBoundaryType[i] = 1;

		  for (int i = 0; i < 15; ++i)
			  leftBoundaryType[i] = 1;
		  for (int i = 15; i < b1_length+15; ++i)
			  leftBoundaryType[i] = 0;
		  for (int i = b1_length; i < b3_length; ++i)
			  leftBoundaryType[i] = 1;

		  for (int i = 0; i < b3_width; ++i)
			  bottomBoundaryType[i] = 1;

		  for (int i = 0; i < b3_length; ++i)
			  rightBoundaryType[i] = 1;


		  double* topBlockBoundary = b3->getTopBlockBoundary();
		  double* leftBlockBoundary = b3->getLeftBlockBoundary();
		  double* bottomBlockBoundary = b3->getBottomBlockBoundary();
		  double* rightBlockBoundary = b3->getRightBlockBoundary();

		  double* topExternalBoundary = b3->getTopExternalBoundary();
		  double* leftExternalBoundary = b3->getLeftExternalBoundary();
		  double* bottomExternalBoundary = b3->getBottomExternalBoundary();
		  double* rightExternalBoundary = b3->getRightExternalBoundary();

		  //printf("\n\n******1-1\n\n");

		  for (int i = 0; i < b3_width; ++i) {
			  topBlockBoundary[i] = 0;
			  bottomBlockBoundary[i] = 0;

			  topExternalBoundary[i] = 100;
			  bottomExternalBoundary[i] = 10;
		  }

		  for (int i = 0; i < b3_length; ++i) {
			  leftBlockBoundary[i] = 0;
			  rightBlockBoundary[i] = 0;

			  leftExternalBoundary[i] = 10;
			  rightExternalBoundary[i] = 10;
		  }

		  //printf("\n\n******1-2\n\n");

		  Interconnect* i4 = new Interconnect(1, 3, 0, 0, b1_length, NULL, leftExternalBoundary + 15);
		  Interconnect* i5 = new Interconnect(3, 1, 0, 0, b1_length, leftBlockBoundary + 15, NULL);

		  //printf("\n\n******1-3\n\n");

		  for (int i = 0; i < repeatCount; ++i) {
			b3->courted();
			b3->prepareData();

			//printf("\n\n******1-4\n\n");

			MPI_Barrier( MPI_COMM_WORLD );

			//printf("\n\nbefore #3 %d\n\n", i);
			i4->sendRecv(world_rank);
			//printf("\n\nbetween #3\n\n");
			i5->sendRecv(world_rank);
			//printf("\n\nafter #3 %d\n\n", i);

			MPI_Barrier( MPI_COMM_WORLD );
		  }

		  double** resault = b3->getResault();

		  for (int i = 0; i < b3_length; ++i)
			  MPI_Send(resault[i], b3_width, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
	  }

	  MPI_Finalize();
}
