#include <math.h>
#include <stdlib.h>

//важный момент: оптимизировать только скорость
//много кода - не страшно, т.к. генерируется автоматически по одному шаблону

#define CELLSIZE 2

#define StepX 0.1
#define StepY 0.1
#define StepZ 0.1

#define dx2 0.01
#define dy2 0.01
#define dz2 0.01

#define Block0StrideZ 10000
#define Block0StrideY 100
#define Block0StrideX 1

#define Block0CountZ 100
#define Block0CountY 100
#define Block0CountX 100

#define Block0OffsetX 0.0
#define Block0OffsetY 0.0
#define Block0OffsetZ 0.0

#define Block1StrideZ 100
#define Block1StrideY 10
#define Block1StrideX 1

#define Block1CountZ 10
#define Block1CountY 10
#define Block1CountX 10

#define Block1OffsetX 10.0
#define Block1OffsetY 5.0
#define Block1OffsetZ 5.0


//начальные условия - только на CPU
typedef void (*initfunc_ptr_t)( double*, double, double, double );

void Initial0(double* cellstart, double x, double y, double z){
    cellstart[0] = 15.0;
    cellstart[1] = sin(x)*cos(y);    
}

void Initial1(double* cellstart, double x, double y, double z){
    cellstart[0] = 200.0;
    cellstart[1] = 100.0;    
}

//Заполняет result[idx] начальной функцией с номером из initType[idx]
void BlockFillInitialValues(double* result, int* initType,
                            int BlockCountX, int BlockCountY, int BlockCountZ,                            
                            int BlockOffsetX, int BlockOffsetY, int BlockOffsetZ){    

    initfunc_ptr_t initFuncArray[2];  
    initFuncArray[0] = Initial0;
    initFuncArray[1] = Initial1;
    
    for(int idxZ = 0; idxZ<BlockCountZ; idxZ++)
        for(int idxY = 0; idxY<BlockCountY; idxY++)
            for(int idxX = 0; idxX<BlockCountX; idxX++){
                int idx = (idxZ*BlockCountY*BlockCountX + idxY*BlockCountX + idxX)*CELLSIZE;
                int type = initType[idx];
                initFuncArray[type](result+idx, BlockOffsetX + idxX*StepX, BlockOffsetY + idxY*StepY, BlockOffsetZ + idxZ*StepZ);
            }

}


//граничные условия
//функции типа дирихле для всех границ всех блоков можно делать одни и те же ,
//а один и тот же Нейман на разных границах разных блоков будет отдельной функцией, т.к. придумывает 
//несуществующую точку в своем направлении и с разными stride

typedef void (*boundfunc_ptr_t)( double*, double*, double ); 

//1. Дирихле
void DirichletBound0(double* result, double* source, double t){
    result[0] = 15.0;
    result[1] = sin(t);
}

//2. Нейман
//default zero flow for every side (0-5)
//Блок0
//сторона x=0
void Block0DefaultNeumannBound0(double* result, double* source, double t){           
    result[0] = dx2*(source[Block0StrideX*CELLSIZE] + source[Block0StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE] + source[-Block0StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE] + source[-Block0StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block0StrideX*CELLSIZE + 1] + source[Block0StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE + 1] + source[-Block0StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE + 1] + source[-Block0StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона x=xmax
void Block0DefaultNeumannBound1(double* result, double* source, double t){           
    result[0] = dx2*(source[-Block0StrideX*CELLSIZE] + source[-Block0StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE]  + source[-Block0StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE]  + source[-Block0StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[-Block0StrideX*CELLSIZE + 1] + source[-Block0StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE + 1]  + source[-Block0StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE + 1]  + source[-Block0StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона y=0
void Block0DefaultNeumannBound2(double* result, double* source, double t){           
    result[0] = dx2*(source[Block0StrideX*CELLSIZE] + source[-Block0StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE] + source[Block0StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE] + source[-Block0StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block0StrideX*CELLSIZE + 1] + source[-Block0StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE + 1] + source[Block0StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE + 1] + source[-Block0StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона y=ymax
void Block0DefaultNeumannBound3(double* result, double* source, double t){           
    result[0] = dx2*(source[Block0StrideX*CELLSIZE]  + source[-Block0StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(-source[Block0StrideY*CELLSIZE] + source[-Block0StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE]  + source[-Block0StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block0StrideX*CELLSIZE + 1]  + source[-Block0StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(-source[Block0StrideY*CELLSIZE + 1] + source[-Block0StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE + 1]  + source[-Block0StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона z=0
void Block0DefaultNeumannBound4(double* result, double* source, double t){           
    result[0] = dx2*(source[Block0StrideX*CELLSIZE] + source[-Block0StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE] + source[-Block0StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE] + source[Block0StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block0StrideX*CELLSIZE + 1] + source[-Block0StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE + 1] + source[-Block0StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block0StrideZ*CELLSIZE + 1] + source[Block0StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона z=zmax
void Block0DefaultNeumannBound5(double* result, double* source, double t){           
    result[0] = dx2*(source[Block0StrideX*CELLSIZE]  + source[-Block0StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE]  + source[-Block0StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[-Block0StrideZ*CELLSIZE] + source[-Block0StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block0StrideX*CELLSIZE + 1]  + source[-Block0StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block0StrideY*CELLSIZE + 1]  + source[-Block0StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[-Block0StrideZ*CELLSIZE + 1] + source[-Block0StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}


//Блок1
//сторона x=0
void Block1DefaultNeumannBound0(double* result, double* source, double t){           
    result[0] = dx2*(source[Block1StrideX*CELLSIZE] + source[Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE] + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1] + source[Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE + 1] + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона x=xmax
void Block1DefaultNeumannBound1(double* result, double* source, double t){           
    result[0] = dx2*(source[-Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE]  + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE]  + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[-Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1]  + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE + 1]  + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона y=0
void Block1DefaultNeumannBound2(double* result, double* source, double t){           
    result[0] = dx2*(source[Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE] + source[Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE] + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE + 1] + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона y=ymax
void Block1DefaultNeumannBound3(double* result, double* source, double t){           
    result[0] = dx2*(source[Block1StrideX*CELLSIZE]  + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(-source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE]  + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1]  + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(-source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE + 1]  + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона z=0
void Block1DefaultNeumannBound4(double* result, double* source, double t){           
    result[0] = dx2*(source[Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE] + source[Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE + 1] + source[Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}

//сторона z=zmax
void Block1DefaultNeumannBound5(double* result, double* source, double t){           
    result[0] = dx2*(source[Block1StrideX*CELLSIZE]  + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE]  + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[-Block1StrideZ*CELLSIZE] + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1]  + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1]  + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[-Block1StrideZ*CELLSIZE + 1] + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}


//явно заданное граничное условие Нейманна живет на границе z=zmax (у второго блока, Block1) 
//source и result передаются уже со сдвигом на первый элемент 
void Block1NeumannBound0(double* result, double* source, double t){       
    double bound_value; 
    double nonexistent;        
    bound_value = -10.0;
    nonexistent = source[-Block1StrideZ*CELLSIZE] - 2.0 * bound_value * dz2;
    result[0] = dx2*(source[Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(nonexistent                    + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    bound_value = cos(t);
    nonexistent = source[-Block1StrideZ*CELLSIZE + 1] - 2.0 * bound_value * dz2;
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(nonexistent                        + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}


void getBoundFuncArray(boundfunc_ptr_t** ppBoundFuncs){
    boundfunc_ptr_t* pBoundFuncs = *ppBoundFuncs;
    pBoundFuncs = (boundfunc_ptr_t*) malloc( ( 1 + 6 + 7 ) * sizeof(boundfunc_ptr_t) );        
    
    pBoundFuncs[0] = DirichletBound0;
    
    pBoundFuncs[1] = Block0DefaultNeumannBound0;
    pBoundFuncs[2] = Block0DefaultNeumannBound1;
    pBoundFuncs[3] = Block0DefaultNeumannBound2;
    pBoundFuncs[4] = Block0DefaultNeumannBound3;
    pBoundFuncs[5] = Block0DefaultNeumannBound4;
    pBoundFuncs[6] = Block0DefaultNeumannBound5;
    
    pBoundFuncs[7] = Block1DefaultNeumannBound0;
    pBoundFuncs[8] = Block1DefaultNeumannBound1;
    pBoundFuncs[9] = Block1DefaultNeumannBound2;
    pBoundFuncs[10] = Block1DefaultNeumannBound3;
    pBoundFuncs[11] = Block1DefaultNeumannBound4;
    pBoundFuncs[12] = Block1DefaultNeumannBound5;
    
    pBoundFuncs[13] = Block1NeumannBound0;
}

void releaseBoundFuncArray(boundfunc_ptr_t* BoundFuncs){
    free(BoundFuncs);    
}


//основная функция
void Block0MainFunc(double* result, 
                    double* source,  //исходный X
                    double time,                
                    char** borderTypes,  // 0 = Дирихле или Нейман, 1 = склейка
                    int** borders,      // номер соответствующего условия,
                                        // функции из набора Дирихле и Неймана
                                        // или набора интерконнектов (набор у каждого блока свой,
                                        // чтобы не плодить, а список функций один для всех)
                    double** ic,
                    boundfunc_ptr_t* pBoundFuncs.
                    double* params    ){

    if (idxZ == 0) 
        if ( (idxY>0)&&(idxX>0)&&(idxY<countY-1)&&(idxX<countX-1) ){            
            cellStart2D = idxY*countX + idxX;
            for (int idx = 0; idx<cellSize; idx++)
                if (borderTypes[BD_ZSTART][cellstart2D+idx] == BT_DIRICHLET)
                    result[cellstart+idx] = DirichletBounds[ borders[BD_ZSTART][cellstart2D+idx] ] ();
                else if (borderTypes[BD_ZSTART][cellstart2D+idx] == BT_NEUMANN)
                    result[cellstart+idx] = NeumannBounds[ borders[BD_ZSTART][cellstart2D+idx] ] (source,);    
                else //interconnect
                    result[cellstart+idx] = 
        }
    if (idxZ == countZ-1) 
    if (idxY == 0) 
    if (idxY == countY-1) 
    if (idxX == 0) 
    if (idxX == countX-1) 

    int cellIdx = Block0CountY*Block0CountX;
    for(int idxZ = 1; idxZ<Block0CountX-1; idxZ++){
        cellIdx+=Block0CountX;
        for(int idxY = 1; idxY<Block0CountY-1; idxY++){
            cellIdx+=1;   
            for(int idxX = 1; idxX<Block0CountX-1; idxX++){
                result[cellIdx*CELLSIZE+0] = 1 +  params
                      dx2*(source[(cellIdx+Block1StrideX)*CELLSIZE]  + source[-Block1StrideX*CELLSIZE] - 2.0*source[cellIdx*CELLSIZE+0]) //вторая по x
                    + dy2*(source[Block1StrideY*CELLSIZE]  + source[-Block1StrideY*CELLSIZE] - 2.0*source[cellIdx*CELLSIZE+0]) //вторая по y
                    + dz2*(source[Block1StrideZ*CELLSIZE] + source[-Block1StrideZ*CELLSIZE] - 2.0*source[cellIdx*CELLSIZE+0]);//вторая по z
                result[cellIdx*CELLSIZE+1] =
                      dx2*(source[Block1StrideX*CELLSIZE + 1]  + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[cellIdx*CELLSIZE+1]) //вторая по x
                    + dy2*(source[Block1StrideY*CELLSIZE + 1]  + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[cellIdx*CELLSIZE+1]) //вторая по y
                    + dz2*(source[Block1StrideZ*CELLSIZE + 1] + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[cellIdx*CELLSIZE+1]);//вторая по z

            }    
            cellIdx+=1;   
        }
        cellIdx+=Block0CountX;
    }
}

