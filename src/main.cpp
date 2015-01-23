#include "BlockCpu.h"
#include "Interconnect.h"

#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "Domain.h"

int main(int argc, char * argv[]) {

	// Initialize the MPI environment
	  MPI_Init(NULL, NULL);

	  // Get the number of processes
	  int world_size;
	  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	  // Get the rank of the process
	  int world_rank;
	  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	  int repeatCount = atoi(argv[1]);

	  Domain *d = new Domain(world_rank, world_size, 4, 3);

	  for (int i = 0; i < repeatCount; ++i) {
		  d->calc(world_rank, 4, 3);
	  }
	  d->print(world_rank, 4);

	  /*int b0_length = 50;
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

		  int* topBorderType = b0->getTopBorderType();
		  int* leftBorderType = b0->getLeftBorderType();
		  int* bottomBorderType = b0->getBottomBorderType();
		  int* rightBorderType = b0->getRightBorderType();

		  for (int i = 0; i < b0_width; ++i)
			  topBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b0_length; ++i)
			  leftBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b0_width; ++i)
			  bottomBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < 15; ++i)
			  rightBorderType[i] = BY_FUNCTION;
		  for (int i = 15; i < b1_length+15; ++i)
			  rightBorderType[i] = BY_ANOTHER_BLOCK;
		  for (int i = b1_length+15; i < b0_length; ++i)
			  rightBorderType[i] = BY_FUNCTION;


		  double* topBlockBorder = b0->getTopBlockBorder();
		  double* leftBlockBorder = b0->getLeftBlockBorder();
		  double* bottomBlockBorder = b0->getBottomBlockBorder();
		  double* rightBlockBorder = b0->getRightBlockBorder();

		  double* topExternalBorder = b0->getTopExternalBorder();
		  double* leftExternalBorder = b0->getLeftExternalBorder();
		  double* bottomExternalBorder = b0->getBottomExternalBorder();
		  double* rightExternalBorder = b0->getRightExternalBorder();

		  //printf("\n\n******0-1\n\n");

		  for (int i = 0; i < b0_width; ++i) {
			  topBlockBorder[i] = 0;
			  bottomBlockBorder[i] = 0;

			  topExternalBorder[i] = 100 * cos( (i - 12) / 9. );
			  bottomExternalBorder[i] = 10;
		  }

		  for (int i = 0; i < b0_length; ++i) {
			  leftBlockBorder[i] = 0;
			  rightBlockBorder[i] = 0;

			  leftExternalBorder[i] = 10;
			  rightExternalBorder[i] = 10;
		  }

		  //printf("\n\n******0-2\n\n");

		  Interconnect* i2 = new Interconnect(0, 1, 0, 0, b1_length, rightBlockBorder + 15, NULL);
		  Interconnect* i3 = new Interconnect(1, 0, 0, 0, b1_length, NULL, rightExternalBorder + 15);

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

		  int* topBorderType = b1->getTopBorderType();
		  int* leftBorderType = b1->getLeftBorderType();
		  int* bottomBorderType = b1->getBottomBorderType();
		  int* rightBorderType = b1->getRightBorderType();


		  for (int i = 0; i < 15; ++i)
			  topBorderType[i] = BY_FUNCTION;
		  for (int i = 15; i < b2_width + 15; ++i)
			  topBorderType[i] = BY_ANOTHER_BLOCK;
		  for (int i = b2_width + 15; i < b1_width; ++i)
			  topBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b1_length; ++i)
			  leftBorderType[i] = BY_ANOTHER_BLOCK;

		  for (int i = 0; i < b1_width; ++i)
			  bottomBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b1_length; ++i)
			  rightBorderType[i] = BY_ANOTHER_BLOCK;


		  double* topBlockBorder = b1->getTopBlockBorder();
		  double* leftBlockBorder = b1->getLeftBlockBorder();
		  double* bottomBlockBorder = b1->getBottomBlockBorder();
		  double* rightBlockBorder = b1->getRightBlockBorder();

		  double* topExternalBorder = b1->getTopExternalBorder();
		  double* leftExternalBorder = b1->getLeftExternalBorder();
		  double* bottomExternalBorder = b1->getBottomExternalBorder();
		  double* rightExternalBorder = b1->getRightExternalBorder();

		  //printf("\n\n******1-1\n\n");

		  for (int i = 0; i < b1_width; ++i) {
			  topBlockBorder[i] = 0;
			  bottomBlockBorder[i] = 0;

			  topExternalBorder[i] = 100 * cos( (i - 25) / 20. );
			  bottomExternalBorder[i] = 10;
		  }

		  for (int i = 0; i < b1_length; ++i) {
			  leftBlockBorder[i] = 0;
			  rightBlockBorder[i] = 0;

			  leftExternalBorder[i] = 10;
			  rightExternalBorder[i] = 10;
		  }

		  //printf("\n\n******1-2\n\n");


		  Interconnect* i0 = new Interconnect(1, 2, 0, 0, b2_width, topBlockBorder + 15, NULL);
		  Interconnect* i1 = new Interconnect(2, 1, 0, 0, b2_width, NULL, topExternalBorder + 15);
		  Interconnect* i2 = new Interconnect(0, 1, 0, 0, b1_length, NULL, leftExternalBorder);
		  Interconnect* i3 = new Interconnect(1, 0, 0, 0, b1_length, leftBlockBorder, NULL);
		  Interconnect* i4 = new Interconnect(1, 3, 0, 0, b1_length, rightBlockBorder, NULL);
		  Interconnect* i5 = new Interconnect(3, 1, 0, 0, b1_length, NULL, rightExternalBorder);

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

		  int* topBorderType = b2->getTopBorderType();
		  int* leftBorderType = b2->getLeftBorderType();
		  int* bottomBorderType = b2->getBottomBorderType();
		  int* rightBorderType = b2->getRightBorderType();

		  //printf("\n\n******2-0\n\n");
		  for (int i = 0; i < b2_width; ++i)
			  topBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b2_length; ++i)
			  leftBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b2_width; ++i)
			  bottomBorderType[i] = BY_ANOTHER_BLOCK;

		  for (int i = 0; i < b2_length; ++i)
			  rightBorderType[i] = BY_FUNCTION;


		  double* topBlockBorder = b2->getTopBlockBorder();
		  double* leftBlockBorder = b2->getLeftBlockBorder();
		  double* bottomBlockBorder = b2->getBottomBlockBorder();
		  double* rightBlockBorder = b2->getRightBlockBorder();

		  double* topExternalBorder = b2->getTopExternalBorder();
		  double* leftExternalBorder = b2->getLeftExternalBorder();
		  double* bottomExternalBorder = b2->getBottomExternalBorder();
		  double* rightExternalBorder = b2->getRightExternalBorder();

		  //printf("\n\n******2-1\n\n");

		  for (int i = 0; i < b2_width; ++i) {
			  topBlockBorder[i] = 0;
			  bottomBlockBorder[i] = 0;

			  topExternalBorder[i] = 10;
			  bottomExternalBorder[i] = 100 * cos( (i - 25) / 20. );
		  }

		  for (int i = 0; i < b2_length; ++i) {
			  leftBlockBorder[i] = 0;
			  rightBlockBorder[i] = 0;

			  leftExternalBorder[i] = 10;
			  rightExternalBorder[i] = 10;
		  }

		  //printf("\n\n******2-2\n\n");

		  Interconnect* i0 = new Interconnect(1, 2, 0, 0, b2_width, NULL, bottomExternalBorder);
		  Interconnect* i1 = new Interconnect(2, 1, 0, 0, b2_width, bottomBlockBorder, NULL);

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

		  int* topBorderType = b3->getTopBorderType();
		  int* leftBorderType = b3->getLeftBorderType();
		  int* bottomBorderType = b3->getBottomBorderType();
		  int* rightBorderType = b3->getRightBorderType();

		  for (int i = 0; i < b3_width; ++i)
			  topBorderType[i] = 1;

		  for (int i = 0; i < 15; ++i)
			  leftBorderType[i] = BY_FUNCTION;
		  for (int i = 15; i < b1_length+15; ++i)
			  leftBorderType[i] = BY_ANOTHER_BLOCK;
		  for (int i = b1_length; i < b3_length; ++i)
			  leftBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b3_width; ++i)
			  bottomBorderType[i] = BY_FUNCTION;

		  for (int i = 0; i < b3_length; ++i)
			  rightBorderType[i] = BY_FUNCTION;


		  double* topBlockBorder = b3->getTopBlockBorder();
		  double* leftBlockBorder = b3->getLeftBlockBorder();
		  double* bottomBlockBorder = b3->getBottomBlockBorder();
		  double* rightBlockBorder = b3->getRightBlockBorder();

		  double* topExternalBorder = b3->getTopExternalBorder();
		  double* leftExternalBorder = b3->getLeftExternalBorder();
		  double* bottomExternalBorder = b3->getBottomExternalBorder();
		  double* rightExternalBorder = b3->getRightExternalBorder();

		  //printf("\n\n******1-1\n\n");

		  for (int i = 0; i < b3_width; ++i) {
			  topBlockBorder[i] = 0;
			  bottomBlockBorder[i] = 0;

			  topExternalBorder[i] = 100 * cos( (i - 12) / 9. );
			  bottomExternalBorder[i] = 10;
		  }

		  for (int i = 0; i < b3_length; ++i) {
			  leftBlockBorder[i] = 0;
			  rightBlockBorder[i] = 0;

			  leftExternalBorder[i] = 10;
			  rightExternalBorder[i] = 10;
		  }

		  //printf("\n\n******1-2\n\n");

		  Interconnect* i4 = new Interconnect(1, 3, 0, 0, b1_length, NULL, leftExternalBorder + 15);
		  Interconnect* i5 = new Interconnect(3, 1, 0, 0, b1_length, leftBlockBorder + 15, NULL);

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
	  }*/

	  MPI_Finalize();
}
