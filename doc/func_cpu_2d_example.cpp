#include <math.h>
#include <stdlib.h>

#define CELLSIZE 2

#define DX 0.1
#define DY 0.1

#define DX2 0.01
#define DY2 0.01

#define Block0StrideX 1
#define Block0StrideY 100

#define Block0CountX 100
#define Block0CountY 100

#define Block0OffsetX 0.0
#define Block0OffsetY 0.0

//===================НАЧАЛЬНЫЕ УСЛОВИЯ==========================//
//начальные условия - только на CPU
typedef void (*initfunc2d_ptr_t)( double* result, int idxX, int idxY);
typedef void (*initfunc2d_fill_ptr_t)( double* result, int* initType);

//для каждого блока свой набор точечных начальных функций и одна функция-заполнитель
void Block0Initial0(double* result, int idxX, int idxY){    
    double x = Block0OffsetX + idxX*DX;
    double y = Block0OffsetY + idxY*DY;
    int idx = (idxY*Block0CountX + idxX)*CELLSIZE;
    result[idx] = 15.0;
    result[idx+1] = sin(x)*cos(y);    
}

void Block0Initial1(double* result, int idxX, int idxY){
    double x = Block0OffsetX + idxX*DX;
    double y = Block0OffsetY + idxY*DY;
    int idx = (idxY*Block0CountX + idxX)*CELLSIZE;
    result[idx] = 200.0;
    result[idx+1] = 100.0;    
}

//Заполняет result[idx] начальной функцией с номером из initType[idx]
void Block0FillInitialValues(double* result, int* initType){
    initfunc2d_ptr_t initFuncArray[2];
    initFuncArray[0] = Block0Initial0;
    initFuncArray[1] = Block0Initial1;
    for(int idxY = 0; idxY<Block0CountY; idxY++)
        for(int idxX = 0; idxX<Block0CountX; idxX++){
            int idx = (idxY*Block0CountX + idxX)*CELLSIZE;
            int type = initType[idx];
            initFuncArray[type](result, idxX, idxY);
        }
}


//Функции-заполнители нужно собрать в массив и отдать домену
void getInitFuncArray(initfunc2d_fill_ptr_t** ppInitFuncs){
    initfunc2d_fill_ptr_t* pInitFuncs = *ppInitFuncs;
    pInitFuncs = (initfunc2d_fill_ptr_t*) malloc( 1 * sizeof(initfunc2d_fill_ptr_t) );            
    pInitFuncs[0] = Block0FillInitialValues;   
}

void releaseInitFuncArray(initfunc2d_fill_ptr_t* InitFuncs){
    free(InitFuncs);    
}


//===================ГРАНИЧНЫЕ УСЛОВИЯ==========================//
//функции типа дирихле для всех границ всех блоков можно делать одни и те же ,
//а один и тот же Нейман на разных границах разных блоков будет отдельной функцией, т.к. придумывает 
//несуществующую точку в своем направлении и с разными stride
typedef void (*func2d_ptr_t)(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic);

//Основная функция
void Block0CentralFunction(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){       
    int idx = ( idxY * Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx+Block0StrideX*CELLSIZE] + source[idx-Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx+Block0StrideY*CELLSIZE] + source[idx-Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx+Block0StrideX*CELLSIZE + 1] + source[idx-Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx+Block0StrideY*CELLSIZE + 1] + source[idx-Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//условия по умолчанию для каждой грани (6 штук),
//для каждого ребра (12 штук) и для каждой вершины (8 штук)
//Блок0
//y=0, x=0
void Block0DefaultNeumannBound0(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){
    int idx = ( Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx+Block0StrideX*CELLSIZE] + source[idx+Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx+Block0StrideY*CELLSIZE] + source[idx+Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx+Block0StrideX*CELLSIZE + 1] + source[idx+Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx+Block0StrideY*CELLSIZE + 1] + source[idx+Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//сторона y=0, x центральные
void Block0DefaultNeumannBound1(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){       
    int idx = ( idxY * Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx+Block0StrideX*CELLSIZE] + source[idx-Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx+Block0StrideY*CELLSIZE] + source[idx+Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx+Block0StrideX*CELLSIZE + 1] + source[idx-Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx+Block0StrideY*CELLSIZE + 1] + source[idx+Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//сторона y=0, x=xmax
void Block0DefaultNeumannBound2(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){       
    int idx = ( idxY * Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx-Block0StrideX*CELLSIZE] + source[idx-Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx+Block0StrideY*CELLSIZE] + source[idx+Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx-Block0StrideX*CELLSIZE + 1] + source[idx-Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx+Block0StrideY*CELLSIZE + 1] + source[idx+Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//y центральные, x=0
void Block0DefaultNeumannBound3(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){
    int idx = ( Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx+Block0StrideX*CELLSIZE] + source[idx+Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx+Block0StrideY*CELLSIZE] + source[idx-Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx+Block0StrideX*CELLSIZE + 1] + source[idx+Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx+Block0StrideY*CELLSIZE + 1] + source[idx-Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//y=центральные, x=xmax
void Block0DefaultNeumannBound4(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){       
    int idx = ( idxY * Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx-Block0StrideX*CELLSIZE] + source[idx-Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx+Block0StrideY*CELLSIZE] + source[idx-Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx-Block0StrideX*CELLSIZE + 1] + source[idx-Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx+Block0StrideY*CELLSIZE + 1] + source[idx-Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//сторона y=ymax, x=0
void Block0DefaultNeumannBound5(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){
    int idx = ( Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx+Block0StrideX*CELLSIZE] + source[idx+Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx-Block0StrideY*CELLSIZE] + source[idx-Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx+Block0StrideX*CELLSIZE + 1] + source[idx+Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx-Block0StrideY*CELLSIZE + 1] + source[idx-Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//сторона y=ymax, x центральные
void Block0DefaultNeumannBound6(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){       
    int idx = ( idxY * Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx+Block0StrideX*CELLSIZE] + source[idx-Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx-Block0StrideY*CELLSIZE] + source[idx-Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx+Block0StrideX*CELLSIZE + 1] + source[idx-Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx-Block0StrideY*CELLSIZE + 1] + source[idx-Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}

//сторона y=ymax, x=xmax
void Block0DefaultNeumannBound7(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){
    int idx = ( idxY * Block0StrideY + idxX) * CELLSIZE;
    result[idx]  = 1.0 + source[idx]*source[idx]*source[idx+1] - params[1]*source[idx] + params[0] * (
                 + dx2*(source[idx-Block0StrideX*CELLSIZE] + source[idx-Block0StrideX*CELLSIZE] - 2.0*source[idx]) 
                 + dy2*(source[idx-Block0StrideY*CELLSIZE] + source[idx-Block0StrideY*CELLSIZE] - 2.0*source[idx]) );
    result[idx+1] =  params[2] * source[idx] - source[idx] * source[idx] * source[idx+1] + params[0] * (
                  + dx2*(source[idx-Block0StrideX*CELLSIZE + 1] + source[idx-Block0StrideX*CELLSIZE + 1] - 2.0*source[idx+1])
                  + dy2*(source[idx-Block0StrideY*CELLSIZE + 1] + source[idx-Block0StrideY*CELLSIZE + 1] - 2.0*source[idx+1]) );
}




//явно заданное граничное условие Нейманна живет на границе z=zmax (у второго блока, Block1) 
//source и result передаются уже со сдвигом на первый элемент 
void Block0NeumannBound0(double* result, double* source, double t, int idxX, int idxY, double* params, double** ic){
    double bound_value; 
    double nonexistent;        
    bound_value = -10.0;
    nonexistent =    source[-Block1StrideZ*CELLSIZE] - 2.0 * bound_value * dz2;
    result[0] = dx2*(source[Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(nonexistent                    + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]);//вторая по z
    bound_value = cos(t);
    nonexistent =    source[-Block1StrideZ*CELLSIZE + 1] - 2.0 * bound_value * dz2;
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(nonexistent                        + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]);//вторая по z
}


//interconnects
void IcFuncZstart(double* result, double* source, double t, double* params, double* border ){  
    result[0] = dx2*(source[Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE] + border[0]                       - 2.0*source[0]);//вторая по z
    result[1] = dx2*(source[Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
              + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
              + dz2*(source[Block1StrideZ*CELLSIZE + 1] + border[0]                           - 2.0*source[1]);//вторая по z
}



void DirichletBound0(double* result, double* source, double t){
    result[0] = 15.0;
    result[1] = sin(t);
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
    
    pBoundFuncs[13] = Block1Bound0;
}

void releaseBoundFuncArray(boundfunc_ptr_t* BoundFuncs){
    free(BoundFuncs);    
}


void Block0InnerFunc0(double* result, double* source, double t, double* params){
    result[0] = 1.0 + source[0]*source[0]*source[1] - params[1]*source[0] + params[0] * (  
           dx2*(source[Block1StrideX*CELLSIZE] + source[-Block1StrideX*CELLSIZE] - 2.0*source[0]) //вторая по x
         + dy2*(source[Block1StrideY*CELLSIZE] + source[-Block1StrideY*CELLSIZE] - 2.0*source[0]) //вторая по y
         + dz2*(source[Block1StrideZ*CELLSIZE] + source[-Block1StrideZ*CELLSIZE] - 2.0*source[0]) );//вторая по z
    result[1] = params[2]*source[0] - source[0]*source[0]*source[1] + params[0] * (
           dx2*(source[Block1StrideX*CELLSIZE + 1] + source[-Block1StrideX*CELLSIZE + 1] - 2.0*source[1]) //вторая по x
         + dy2*(source[Block1StrideY*CELLSIZE + 1] + source[-Block1StrideY*CELLSIZE + 1] - 2.0*source[1]) //вторая по y
         + dz2*(source[Block1StrideZ*CELLSIZE + 1] + source[-Block1StrideZ*CELLSIZE + 1] - 2.0*source[1]) );//вторая по z
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
                    func_ptr_t* pFuncs.
                    double* params    ){
      
    int cellStart = 0;
    //layer z = 0  
    for(int idxY = 0; idxY<Block0CountY; idxY++)
        for(int idxX = 0; idxX<Block0CountX; idxX++){
            ind idx2d = idxY*Block1StrideY+idxX;
            if (borderTypes[BD_ZSTART][idx2d] == 0)
                pBoundFuncs[ borders[BD_ZSTART][idx2d] ](result+cellStart, source+cellStart, time, params);
            else
                IcFuncZstart(result+cellStart, source+cellStart, time, params, ic[BD_ZSTART]+idx2d*CELLSIZE);
            cellStart+= CELLSIZE;
        }
    
    //internal z layers    
    for(int idxZ = 1; idxZ<Block0CountZ-1; idxZ++){
        //line y=0
        cellStart += Block0CountX * CELLSIZE;
        //internal y lines
        for(int idxY = 1; idxY<Block0CountY-1; idxY++){
            //point x = 0   
            cellStart += CELLSIZE;
            //internal x points
            for(int idxX = 1; idxX<Block0CountX-1; idxX++)
                Block0InnerFunc0(result+cellStart, source+cellStart, time, params);
            cellStart+= CELLSIZE;   
            //point x = xmax-1
        }
        //line y=ymax-1
        cellIdx+=Block0CountY*Block0CountX * CELLSIZE;
    }        
    //layer z = zmax-1  
    
    
    
}

