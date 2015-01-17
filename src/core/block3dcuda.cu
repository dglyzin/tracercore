/*
 * block3dcpu.cpp
 *
 *  Created on: Nov 5, 2014
 *      Author: dglyzin
 */
#include <cstring>
#include <stdlib.h>
#include "mystdint.h"
#include "block3dcuda.h"
#include "approx.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>


EXPORT_XX
int getCudaDevCount(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}
// ******* LOADING ROUTINES **********
void Block3dCuda::SetSoilData(int soilCount, float* frozenSoilConductivity, float* thawedSoilConductivity,
                               float* drySoilSpecHeat, float* frozenSoilVolHeat, float* thawedSoilVolHeat,
                               float* drySoilDensity, float* waterPerDrySoilMass, float* phaseTransitionPoint,
                               int* liquidWaterGraphLength, float* liquidWaterGraphs,
                               float* temperatureGraph, float* hMelt, float* liqPartAtTr,
                               float waterLatentHeat){
}


void Block3dCuda::SetCavernData(int cavernCount, int* cavernBound){
  
}

void Block3dCuda::SetBoundData(int boundCount, int *boundTypes, float *temperatureLag,
                     float *alphaConvection, float *boundValues){
  
}



void Block3dCuda::SetHeatSourceData(int heatSourceCount, float *heatSourceDensity){
    
}


void Block3dCuda::PrepareEngine(float timeStep){
 /*   int volumescount = mXcount*mYcount*mZcount;
    mFlowXSize =(mXcount+1)* mYcount *mZcount;
    mFlowYSize = mXcount*(mYcount+1)*mZcount;
    mFlowZSize = mXcount*mYcount*(mZcount+1);

    cudaMalloc((void**)&mXsizesDev, mXcount*sizeof(float));
    cudaMalloc((void**)&mYsizesDev, mYcount*sizeof(float));
    cudaMalloc((void**)&mZsizesDev, mZcount*sizeof(float));
    cudaMemcpy(mXsizesDev, mXsizes, mXcount*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mYsizesDev, mYsizes, mYcount*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(mZsizesDev, mZsizes, mZcount*sizeof(float), cudaMemcpyHostToDevice);

    mHeatFlowX = (float*) malloc( mFlowXSize*sizeof(float));
    mHeatFlowY = (float*) malloc( mFlowYSize*sizeof(float));
    mHeatFlowZ = (float*) malloc( mFlowZSize*sizeof(float));
    mRecConductivity = (float*) malloc( volumescount*sizeof(float));

    cudaMalloc((void**)&mHeatFlowXDev, mFlowXSize*sizeof(float));
    cudaMalloc((void**)&mHeatFlowYDev, mFlowYSize*sizeof(float));
    cudaMalloc((void**)&mHeatFlowZDev, mFlowZSize*sizeof(float));
    cudaMalloc((void**)&mRecConductivityDev, volumescount*sizeof(float));
*/
    
    //create dim3 params for kernel launch
    int gridx, gridy, gridz;

    mCuBlockSize = dim3(16,8,1);

    gridx = mXcount / mCuBlockSize.x;
    if (mXcount % mCuBlockSize.x)
        gridx++;
    gridy = mYcount / mCuBlockSize.y;
    if (mYcount % mCuBlockSize.y)
        gridy++;
    gridz = mZcount / mCuBlockSize.z;
    if (mZcount % mCuBlockSize.z)
        gridz++;
    mCuVolGridSize = dim3(gridx, gridy, gridz);

    mCuSurfBlockSize = dim3(16,8,1);

    gridx = mXcount / mCuSurfBlockSize.x;
    if (mXcount % mCuSurfBlockSize.x)
        gridx++;
    gridy = mYcount / mCuSurfBlockSize.y;
    if (mYcount % mCuSurfBlockSize.y)
        gridy++;
    mCuSurfGridSizeZ = dim3(gridx, gridy, 1);

    gridx = mXcount / mCuSurfBlockSize.x;
    if (mXcount % mCuSurfBlockSize.x)
        gridx++;
    gridy = mZcount / mCuSurfBlockSize.y;
    if (mZcount % mCuSurfBlockSize.y)
        gridy++;
    mCuSurfGridSizeY = dim3(gridx, gridy, 1);

    gridx = mYcount / mCuSurfBlockSize.x;
    if (mYcount % mCuSurfBlockSize.x)
        gridx++;
    gridy = mZcount / mCuSurfBlockSize.y;
    if (mZcount % mCuSurfBlockSize.y)
        gridy++;
    mCuSurfGridSizeX = dim3(gridx, gridy, 1);

    //we have enthalpy already - either loaded or precomputed on cpu
    //gpu engine should copy everything here as the next two calls need everything be ready
    ComputeTemperatureAndLiquidPart();

    ComputeRecConductivity();

}


// ************* ITERATIONS, COMPUTATIONS *********************
void Block3dCuda::FillInterconnectBuffer(int idx){
    //fill mIcSourceConductivity[icIdx], mIcSourceTemperature[icIdx] with data to send
        Interconnect *ic = mInterconnects[idx];
        cudaMemcpyKind direction = cudaMemcpyDefault;
        //cudaMemcpyKind direction = cudaMemcpyDeviceToHost;
        if (ic->mSourceSide == ZSTART){
            cudaMemcpy(ic->mSourceTemperature, mTemperatureDev, mZSliceSize*sizeof(float), direction  );
            cudaMemcpy(ic->mSourceConductivity, mRecConductivityDev, mZSliceSize*sizeof(float), direction );

        }
        else if (ic->mSourceSide == ZEND){
            cudaMemcpy(ic->mSourceConductivity, mRecConductivityDev+(mZcount-1)*mZSliceSize, mZSliceSize*sizeof(float), direction );
            cudaMemcpy(ic->mSourceTemperature, mTemperatureDev+(mZcount-1)*mZSliceSize, mZSliceSize*sizeof(float), direction  );
        }
        else
            assert(false);
}


void Block3dCuda::ProcessOneStep(int month) {
    mMonth = month;
    //at this time we know current temperature, enthalpy and reciprocal conductivity
    //Compute Heat Flows
    cudaSetDevice(mLocationDevice-1);
    ComputeHeatFlow();

    //Update Enthalpy
    UpdateEnthalpy();

    //Update Temperature
    ComputeTemperatureAndLiquidPart();

    ComputeRecConductivity();

}

// ****** SAVING *******
void Block3dCuda::StoreData(){
    int volumescount = mXcount*mYcount*mZcount;
    cudaMemcpy(mTemperature, mTemperatureDev, volumescount*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mEnthalpy, mEnthalpyDev,volumescount*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mAtPhaseTransition, mAtPhaseTransitionDev, volumescount*sizeof(char), cudaMemcpyDeviceToHost);
    cudaMemcpy(mLiquidPart, mLiquidPartDev,volumescount*sizeof(float), cudaMemcpyDeviceToHost);
/*
    cudaMemcpy(mHeatFlowX, mHeatFlowXDev, mFlowXSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mHeatFlowY, mHeatFlowYDev, mFlowYSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mHeatFlowZ, mHeatFlowZDev, mFlowZSize*sizeof(float), cudaMemcpyDeviceToHost);
*/
    cudaMemcpy(mRecConductivity, mRecConductivityDev, volumescount*sizeof(float), cudaMemcpyDeviceToHost);
}

// ********* RELEASE *************
void Block3dCuda::ReleaseResources(){
    //free();
 
    
    //Block3d (anchestor) allocated
    ReleaseBasicResources();

}

///CUDA KERNELS


__global__ void ComputeRecConductivityKN(int Xcount, int Ycount, int Zcount, uint16_t* SoilTypesDev,
                                         float* RecConductivityDev, float* LiquidPartDev,
                                         float* ThawedSoilConductivityDev,
                                         float* FrozenSoilConductivityDev ){
    int idxX = threadIdx.x+blockIdx.x*blockDim.x;
    int idxY = threadIdx.y+blockIdx.y*blockDim.y;
    int idxZ = threadIdx.z+blockIdx.z*blockDim.z;

    if( (idxX<Xcount)&&(idxY<Ycount)&&(idxZ<Zcount) ){
      int entindex = idxZ*Ycount*Xcount+idxY*Xcount+idxX;
      int soilindex = SoilTypesDev[entindex];
      RecConductivityDev[entindex] = 1.0f/(LiquidPartDev[entindex]*ThawedSoilConductivityDev[soilindex] +(1-LiquidPartDev[entindex])*FrozenSoilConductivityDev[soilindex]);
    }
}

void Block3dCuda::ComputeRecConductivity(){
    ComputeRecConductivityKN<<<mCuVolGridSize, mCuBlockSize>>>(
                             mXcount, mYcount, mZcount, mSoilTypesDev,
                             mRecConductivityDev, mLiquidPartDev,
                             mThawedSoilConductivity,
                             mFrozenSoilConductivity);  
}


__global__ void UpdateEnthalpyKN(int Xcount, int Ycount, int Zcount, float* HeatFlowZDev,
                                         float* HeatFlowYDev, float* HeatFlowXDev,
                                         float* EnthalpyDev, float TimeStep,
                                         float* XsizesDev,  float* YsizesDev, float* ZsizesDev,
                                         float* HeatSourceDensityDev, uint16_t *HeatSourcesDev,
                                         int Month){
    int idxX = threadIdx.x+blockIdx.x*blockDim.x;
    int idxY = threadIdx.y+blockIdx.y*blockDim.y;
    int idxZ = threadIdx.z+blockIdx.z*blockDim.z;
  
}


void Block3dCuda::UpdateEnthalpy(){
 /*   UpdateEnthalpyKN<<<mCuVolGridSize, mCuBlockSize>>>(
                                         mXcount, mYcount, mZcount, mHeatFlowZDev,
                                         mHeatFlowYDev, mHeatFlowXDev,
                                         mEnthalpyDev, mTimeStep,
                                         mXsizes, mYsizes, mZsizes,
                                         mHeatSourceDensity, mHeatSourcesDev,
                                         mMonth);    */
}


__device__ float GetTempForSoilEntKN(uint16_t soilindex, float enthalpy,
                                     float* TemperatureGraphDev ){

   
    return -0.f;

}


__device__ float GetLiquidPartForSoilTempKN(uint16_t soilindex, float temperature,
                                            float** LiqidWaterGraphPtrsDev,
                                            int* LiquidWaterGraphLengthDev
                                            ){
      return 0.f;
}

__global__ void ComputeTemperatureAndLiquidPartKN(int Xcount, int Ycount, int Zcount,
                                                 uint16_t* SoilTypesDev,   //float WaterLatentHeat,
                                                 float* ThawedSoilVolHeatDev, //float* WaterPerDrySoilMassDev,
                                                 float* PhaseTransitionPointDev, //float* DrySoilSpecHeatDev,
                                                 //float WaterSpecHeat,
                                                 float* EnthalpyDev,
                                                 char* AtPhaseTransitionDev, float* TemperatureDev,
                                                 float* LiquidPartDev,
                                                 float** LiqidWaterGraphPtrsDev,
                                                 int* LiquidWaterGraphLengthDev,
                                                 float* TemperatureGraphDev,
                                                 float* HMeltDev, float* LiqPartAtTrDev
                                                 ){
    int idxX = threadIdx.x+blockIdx.x*blockDim.x;
    int idxY = threadIdx.y+blockIdx.y*blockDim.y;
    int idxZ = threadIdx.z+blockIdx.z*blockDim.z;

    
}


void Block3dCuda::ComputeTemperatureAndLiquidPart(){
    ComputeTemperatureAndLiquidPartKN<<<mCuVolGridSize, mCuBlockSize>>>(
                                        mXcount, mYcount, mZcount,
                                        mSoilTypesDev,
                                        mThawedSoilVolHeat,
                                        mPhaseTransitionPoint,
                                        mEnthalpyDev,
                                        mAtPhaseTransitionDev, mTemperatureDev,
                                        mLiquidPartDev,
                                        mLiqidWaterGraphPtrs,
                                        mLiquidWaterGraphLength,
                                        mTemperatureGraph,
                                        mHMelt, mLiqPartAtTr);    
}

//************** HEAT FLOW *************************************//
//assume direction  is 1, that is bound is at x,y,z =0
__device__ float GetBoundFlowKN(uint16_t bound, int sideMidx, int sideNidx, int volumeIdx, float area, float size,
                                int* BoundTypesDev,
                                float* BoundValuesDev, float* TemperatureDev,
                                float* RecConductivityDev, float* AlphaConvectionDev,
                                float* TemperatureLagDev, int Month,
								int* icMshift, int* icNshift, int* icNdim,
								float** icOwnerTemperature,
								float** icOwnerConductivity,
								float* icSourceH ){
   

    return 0;

}


__global__ void ComputeHeatFlowKN0(int Xcount, int Ycount, int Zcount,
                                   int Xfat,   int Yfat,   int Zfat,
                                   float* NsizesDev, float* MsizesDev, float* RsizesDev,
                                   float* HeatFlow, //result
                                   float* TemperatureDev, float* RecConductivityDev,
                                   uint16_t* Bound0, uint16_t* BoundMax,
                                   int* BoundTypesDev,
                                   float* BoundValuesDev, float* AlphaConvectionDev, float* TemperatureLagDev,
                                   int Month,								   
								   int* icMshift, int* icNshift, int* icNdim,
								   float** icOwnerTemperature,
								   float** icOwnerConductivity,
								   float* icSourceH){
    int idxN = threadIdx.x+blockIdx.x*blockDim.x;
    int idxM = threadIdx.y+blockIdx.y*blockDim.y;

    int idxX, idxY, idxZ, idxR, Mcount, Ncount;
   
}



__global__ void ComputeHeatFlowKN(int Xcount, int Ycount, int Zcount,
                                  int Xfat,   int Yfat,   int Zfat,
                                  float *NsizesDev, //minor non-fat
                                  float *MsizesDev, //major non-fat
                                  float *RsizesDev, //fat
                                  float* HeatFlow, //(Xcount+Xfat)*(Ycount+Yfat)*(Zcount+Zfat)
                                  float* TemperatureDev, //Xcount*Ycount*Zcount
                                  float* RecConductivityDev){
    int idxX = threadIdx.x+blockIdx.x*blockDim.x;
    int idxY = threadIdx.y+blockIdx.y*blockDim.y;
    int idxZ = threadIdx.z+blockIdx.z*blockDim.z;


}



void Block3dCuda::ComputeHeatFlow(){

   
}




