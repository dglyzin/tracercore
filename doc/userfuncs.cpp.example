//this is just an example with appropriate function signatures.
//libuserfuncs.so is regenerated every time a new computation is run

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "userfuncs.h"

#include <iostream>
using namespace std;



#define DX 0.01
#define DX2 0.0001
#define DXM1 100
#define DXM2 10000


#define DY 1
#define DY2 1
#define DYM1 1
#define DYM2 1


#define DZ 1
#define DZ2 1
#define DZM1 1
#define DZM2 1




#define Block0CELLSIZE 1





#define Block0StrideX 1
#define Block0CountX 351
#define Block0OffsetX 0.1

#define Block0StrideY 351
#define Block0CountY 1
#define Block0OffsetY 0

#define Block0StrideZ 351
#define Block0CountZ 1
#define Block0OffsetZ 0


#define PAR_COUNT 2

//===================INITIAL CONDITIONS==========================//




void Initial_0_0(double* cellstart, /*UPDATE*/double t, double x, double y, double z){

        cellstart[0] = 1.0+t;

}



void DirichletInitial_0_1(double* cellstart, /*UPDATE*/double t, double x, double y, double z){
	/*UPDATE*/  //remove: double t = 0;

        cellstart[0] = (0.0);

}





void Block0FillInitialValues(double* result, /*UPDATE*/ double t, unsigned short int* initType){
        initfunc_ptr_t initFuncArray[2];


        initFuncArray[0] = Initial_0_0;



        initFuncArray[1] = DirichletInitial_0_1;



        for(int idxX = 0; idxX<Block0CountX; idxX++){
                int idx = idxX;
                int type = initType[idx];
                initFuncArray[type](result+idx*Block0CELLSIZE, /*UPDATE*/ t, Block0OffsetX + idxX*DX, 0, 0);
        }


}


void getInitFuncArray(initfunc_fill_ptr_t** ppInitFuncs){
        initfunc_fill_ptr_t* pInitFuncs;
        pInitFuncs = (initfunc_fill_ptr_t*) malloc( 1 * sizeof(initfunc_fill_ptr_t) );
        *ppInitFuncs = pInitFuncs;


        pInitFuncs[0] = Block0FillInitialValues;

}

void releaseInitFuncArray(initfunc_fill_ptr_t* InitFuncs){
        free(InitFuncs);
}
//===================PARAMETERS==========================//



void initDefaultParams(double** pparams, int* pparamscount){
        *pparamscount = PAR_COUNT;
        *pparams = (double *) malloc(sizeof(double)*PAR_COUNT);

        (*pparams)[0] = 1;

        (*pparams)[1] = 2;

}
void releaseParams(double *params){
        free(params);
}






//=========================CENTRAL FUNCTIONS FOR BLOCK WITH NUMBER 0========================//

//0 central function for 1d model for block with number 0
void Block0CentralFunction_Eqn0(double* result, double** source, double t, int idxX, int idxY, int idxZ, double* params, double** ic){
         int idx = ( idxX + idxY * Block0StrideY + idxZ * Block0StrideZ) * Block0CELLSIZE;

        // original: U'= a*(b-U(t-1))*U
        result[idx + 0]=params[0]*(params[1]-source[1][idx + 0])*source[0][idx + 0];

}

//device side function pointer declaration and init:
func_ptr_t p_Block0CentralFunction_Eqn0 = Block0CentralFunction_Eqn0;





//=============================BOUNDARY CONDITIONS FOR BLOCK WITH NUMBER 0======================//
//Boundary condition for boundary x = 0
void Block0Dirichlet__side0_bound0__Eqn0(double* result, double** source, double t, int idxX, int idxY, int idxZ, double* params, double** ic){
         int idx = ( idxX + idxY * Block0StrideY + idxZ * Block0StrideZ) * Block0CELLSIZE;

         // original: U'= a*(b-U(t-1))*U

         result[idx + 0]=(0.0);

}
//device side function pointer declaration and init:
func_ptr_t p_Block0Dirichlet__side0_bound0__Eqn0 = Block0Dirichlet__side0_bound0__Eqn0;



//=============================BOUNDARY CONDITIONS FOR BLOCK WITH NUMBER 0======================//
//Boundary condition for boundary x = x_max
void Block0DefaultNeumann__side1__Eqn0(double* result, double** source, double t, int idxX, int idxY, int idxZ, double* params, double** ic){
         int idx = ( idxX + idxY * Block0StrideY + idxZ * Block0StrideZ) * Block0CELLSIZE;

         // original: U'= a*(b-U(t-1))*U

         result[idx + 0]=params[0]*(params[1]-source[1][idx + 0])*source[0][idx + 0];

}
//device side function pointer declaration and init:
func_ptr_t p_Block0DefaultNeumann__side1__Eqn0 = Block0DefaultNeumann__side1__Eqn0;





//===================================FILL FUNCTIONS===========================//



void getBlock0BoundFuncArray(func_ptr_t** ppBoundFuncs){
        func_ptr_t* pBoundFuncs = *ppBoundFuncs;
        pBoundFuncs = (func_ptr_t*) malloc( 3 * sizeof(func_ptr_t) );
        *ppBoundFuncs = pBoundFuncs;

        pBoundFuncs[0] = p_Block0CentralFunction_Eqn0;

        pBoundFuncs[1] = p_Block0Dirichlet__side0_bound0__Eqn0;

        pBoundFuncs[2] = p_Block0DefaultNeumann__side1__Eqn0;

}



void getFuncArray(func_ptr_t** ppBoundFuncs, int blockIdx){

        if (blockIdx == 0)
           getBlock0BoundFuncArray(ppBoundFuncs);

}

void releaseFuncArray(func_ptr_t* BoundFuncs){
        free(BoundFuncs);
}
