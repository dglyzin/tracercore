/*
 * block3dcpu.cpp
 *
 *  Created on: Nov 5, 2014
 *      Author: dglyzin
 */
#include <cstring>
#include <stdlib.h>
#include "mystdint.h"
#include "block3dcpu.h"
#include "approx.h"
#include <assert.h>
#include <omp.h>

// ******* LOADING ROUTINES **********
void Block3dCpu::SetSoilData(int soilCount, float* frozenSoilConductivity, float* thawedSoilConductivity,
                               float* drySoilSpecHeat, float* frozenSoilVolHeat, float* thawedSoilVolHeat,
                               float* drySoilDensity, float* waterPerDrySoilMass, float* phaseTransitionPoint,
                               int* liquidWaterGraphLength, float* liquidWaterGraphs,
                               float* temperatureGraph, float* hMelt, float* liqPartAtTr,
                               float waterLatentHeat){
   
}

void Block3dCpu::SetCavernData(int cavernCount, int* cavernBound){
    
}

void Block3dCpu::SetBoundData(int boundCount, int *boundTypes, float *temperatureLag,
                     float *alphaConvection, float *boundValues){
    
}


void Block3dCpu::SetHeatSourceData(int heatSourceCount, float *heatSourceDensity){
    
}

void Block3dCpu::PrepareEngine(float timeStep){
    
}



// ************* ITERATIONS, COMPUTATIONS *********************
//TODO Optimize! assign different copy methods, avoid switching
void Block3dCpu::FillInterconnectBuffer(int idx){
    //fill mIcSourceConductivity[icIdx], mIcSourceTemperature[icIdx] with data to send
    Interconnect *ic = mInterconnects[idx];
    if (ic->mSourceSide == ZSTART){
        memcpy(ic->mSourceConductivity, mRecConductivity, mZSliceSize*sizeof(float) );
        memcpy(ic->mSourceTemperature, mTemperature, mZSliceSize*sizeof(float) );
    }
    else if (ic->mSourceSide == ZEND){
        memcpy(ic->mSourceConductivity, mRecConductivity+(mZcount-1)*mZSliceSize, mZSliceSize*sizeof(float));
        memcpy(ic->mSourceTemperature, mTemperature+(mZcount-1)*mZSliceSize, mZSliceSize*sizeof(float) );
    }
    else if (ic->mSourceSide == XSTART)
        for (int iz = 0; iz<mZcount; iz++)
            for (int iy = 0; iy<mYcount; iy++){
                int idx_block = iz*mZSliceSize+iy*mXcount+0;
                int idx_surface = iz*mYcount +iy;
                ic->mSourceConductivity[idx_surface] = mRecConductivity[idx_block];
                ic->mSourceTemperature[idx_surface] = mTemperature[idx_block];
            }
    else if (ic->mSourceSide == XEND)
        for (int iz = 0; iz<mZcount; iz++)
            for (int iy = 0; iy<mYcount; iy++){
                int idx_block = iz*mZSliceSize+iy*mXcount+mXcount-1;
                int idx_surface = iz*mYcount +iy;
                ic->mSourceConductivity[idx_surface] = mRecConductivity[idx_block];
                ic->mSourceTemperature[idx_surface] = mTemperature[idx_block];
            }
    else if (ic->mSourceSide == YSTART)
        for (int iz = 0; iz<mZcount; iz++)
            for (int ix = 0; ix<mXcount; ix++){
                int idx_block = iz*mZSliceSize+0*mXcount+ix;
                int idx_surface = iz*mXcount +ix;
                ic->mSourceConductivity[idx_surface] = mRecConductivity[idx_block];
                ic->mSourceTemperature[idx_surface] = mTemperature[idx_block];
             }
    else if (ic->mSourceSide == YEND)
        for (int iz = 0; iz<mZcount; iz++)
            for (int ix = 0; ix<mXcount; ix++){
                int idx_block = iz*mZSliceSize+(mYcount-1)*mXcount+ix;
                int idx_surface = iz*mXcount +ix;
                ic->mSourceConductivity[idx_surface] = mRecConductivity[idx_block];
                ic->mSourceTemperature[idx_surface] = mTemperature[idx_block];
            }
    else
        assert(false);

}


void Block3dCpu::ProcessOneStep(int month) {
    mMonth = month;
    //Compute Heat Flows
    ComputeHeatFlow();
    //Update Enthalpy
    UpdateEnthalpy();
    //Compute Temperature
    ComputeTemperatureAndLiquidPart();
    //Compute conductivity
    ComputeRecConductivity();
}

//compute temperature from Enthalpy
void Block3dCpu::ComputeTemperatureAndLiquidPart(){
#pragma omp parallel
    {
   
    }
}

void Block3dCpu::ComputeRecConductivity(){

#pragma omp parallel
    {
    

    }

}


//size1xsize2 is the surface area, size 3 is the other dimension
                                                 //indices   of receiver!
float Block3dCpu::GetBorderIcFlow(uint16_t icIdx, int sideMidx, int sideNidx, int volumeIdx, float area, float sizeR, char direction){

    float temp1, temp2;
    float flow=0.f;

    Interconnect *ic = mInterconnects[icIdx];
    //the next index is in sender's coordinates and is used for getting data from sender's arrays
    int sideIdx = (sideMidx + ic->mMshift)*ic->mNdim + (sideNidx + ic->mNshift);

    if (direction){  // ==1 means bound closer to zero than volume
     temp1 = ic->mOwnerTemperature[sideIdx]; //source temperature should be copied here already!
     temp2 = mTemperature[volumeIdx];
    }
    else{
     temp2 = ic->mOwnerTemperature[sideIdx];
     temp1 = mTemperature[volumeIdx];
    }

    flow = 2.0f*(temp1-temp2)/(mRecConductivity[volumeIdx]*sizeR + ic->mOwnerConductivity[sideIdx]*ic->mSourceH);

    return flow*area;

}


float Block3dCpu::GetBorderBoundFlow(uint16_t bound, int volumeIdx, float area, float size, char direction){

   return 0.0;
}


//use boundaries and real blocks enthalpy to compute flows
void Block3dCpu::ComputeHeatFlow(){
    //omp_set_num_threads(2);
#pragma omp parallel
    {
  
    }//end parallel
}

//use known heat flow to update Enthalpy
void Block3dCpu::UpdateEnthalpy(){
#pragma omp parallel
    {
    //positive flow direction corresponds to growth of x,y or z
    int entindex, idxX, idxY, idxZ;
    float heatchange;
    #pragma omp for
    for(idxZ=0; idxZ<(int)mZcount; idxZ++)
        for(idxY=0; idxY<(int)mYcount; idxY++)
            for(idxX=0; idxX<(int)mXcount; idxX++){
                heatchange=0;
                entindex = idxZ*mZSliceSize+idxY*mXcount+idxX; //= idxX+idxY*mXcount+idxZ*mYcount*mXcount
                if (mBodyType[entindex]==BT_SOIL){
                    //up (between z,z-1)
                    heatchange+=mHeatFlow[AXISZ][entindex];
                    //down (between z,z+1)
                    heatchange-=mHeatFlow[AXISZ][entindex+mZSliceSize]; //= idxX+idxY*mXcount+(idxZ+1)*mYcount*mXcount
                    //std::cout<<entindex<<std::endl;
                    //mFlowYSliceSize = mXcount*(mYcount+1)*mZcount;
                    //y, y-1
                    heatchange+=mHeatFlow[AXISY][entindex+idxZ*mXcount];  //= idxX + idxY*mXcount + idxZ*mXcount*(mYcount+1)
                    //y, y+1
                    heatchange-=mHeatFlow[AXISY][entindex+(idxZ+1)*mXcount];  //= idxX + (idxY+1)*mXcount + idxZ*mXcount*(mYcount+1)
                    //mFlowXSize =(mXcount+1)* mYcount *mZcount;
                    //x, x-1
                    heatchange+=mHeatFlow[AXISX][entindex + idxY+ idxZ*mYcount];  //= idxX + idxY*(mXcount+1) + idxZ*(mXcount+1)*mYcount
                    //x, x+1
                    heatchange-=mHeatFlow[AXISX][entindex + idxY+ idxZ*mYcount +1];  //= idxX+1 + idxY*(mXcount+1) + idxZ*(mXcount+1)*mYcount
                    //std::cout<<heatchange<<std::endl;
                    mEnthalpy[entindex]+=heatchange*mTimeStep/mSizes[AXISX][idxX]/mSizes[AXISY][idxY]/mSizes[AXISZ][idxZ]+mHeatSourceDensity[12*mHeatSources[entindex]+mMonth];
                }
                else //BT_CAVERN
                    mEnthalpy[entindex] = -10000;
            }
    }
}



// ****** SAVING *******
void Block3dCpu::StoreData(){
    //just a stub for cpu domain - it stores everything in cpu memory
}

// ********* RELEASE *************
void Block3dCpu::ReleaseResources(){
    //free();
    free(mRecConductivity);
    free(mHeatFlow[AXISZ]);
    free(mHeatFlow[AXISY]);
    free(mHeatFlow[AXISX]);


    free(mHeatSourceDensity);

    free(mBoundValues);
    free(mAlphaConvection);
    free(mTemperatureLag);
    free(mBoundTypes);

    free(mLiqPartAtTr);
    free(mHMelt);
    free(mTemperatureGraph);
    free(mLiqidWaterGraphPtrs);

    free(mLiquidWaterGraphs);
    free(mLiquidWaterGraphLength);
    free(mPhaseTransitionPoint);
    free(mWaterPerDrySoilMass);
    free(mDrySoilDensity);
    free(mThawedSoilVolHeat);
    free(mFrozenSoilVolHeat);
    free(mDrySoilSpecHeat);
    free(mThawedSoilConductivity);
    free(mFrozenSoilConductivity);

    ReleaseBasicResources();
}
