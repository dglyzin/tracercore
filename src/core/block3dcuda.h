/*
 * block3dcuda.h
 *
 *  Created on: Dec 5, 2014
 *      Author: dglyzin
 */

#ifndef BLOCK3DCUDA_H_
#define BLOCK3DCUDA_H_

#include "block3d.h"
#include "mystdint.h"

#include <cuda_runtime_api.h>
#include <cuda.h>


class Block3dCuda: public Block3d {
public:
    void SetSoilData(int soilCount, float* frozenSoilConductivity, float* thawedSoilConductivity,
                         float* drySoilSpecHeat, float* frozenSoilVolHeat, float* thawedSoilVolHeat,
                         float* drySoilDensity, float* waterPerDrySoilMass, float* phaseTransitionPoint,
                         int* liquidWaterGraphLength, float* liquidWaterGraphs,
                         float* temperatureGraph, float* hMelt, float* liqPartAtTr,
                         float waterLatentHeat);
    void SetCavernData(int cavernCount, int* cavernBound);
    void SetInterconnectData(int interconnectCount, int* interconnectOwners, int* interconnectSources,
                         int* mInterconnectSourceSide, float* interconnectSourceH, int* interconnectMshift,
                         int* interconnectNshift, int* mIcOwnerNode, int* mIcOwnerDevice, int* mIcSourceNode,
                         int* mIcSourceDevice, int* interconnectMdim, int* interconnectNdim,
                         float** mIcSourceConductivity, float** mIcSourceTemperature,
                         float** mIcOwnerConductivity, float** mIcOwnerTemperature);
    void SetBoundData(int boundCount, int *boundTypes, float *temperatureLag,
                         float *alphaConvection, float *boundValues);
    void SetHeatSourceData(int heatSourceCount, float *heatSourceDensity);
    //Do all preparations after load (including copy to CUDA memory for cuda block)
    void PrepareEngine(float timeStep);
    //ENGINE
    void FillInterconnectBuffer(int idx);
    void ProcessOneStep(int month);
    //make data from devices available to the local domain
    void StoreData();
    void ReleaseResources();

private:
    void  ComputeTemperatureAndLiquidPart();
    void  ComputeRecConductivity();
    void  UpdateEnthalpy();
    void  ComputeHeatFlow();


    //CUDA State and properties arrays that has their counterparts on cpu (derived from Block3d)
    //Every other pointer derived from Block3d points to gpu memory
    float *mXsizesDev, *mYsizesDev, *mZsizesDev;

    float* mHeatFlowXDev;
    float* mHeatFlowYDev;
    float* mHeatFlowZDev;
    float* mRecConductivityDev;
    float* mTemperatureDev;
    float* mEnthalpyDev;
    char* mAtPhaseTransitionDev;
    float* mLiquidPartDev;

    uint16_t* mSoilTypesDev;
    uint16_t* mHeatSourcesDev;
    uint16_t *mBoundX0Dev, *mBoundXMaxDev, *mBoundY0Dev, *mBoundYMaxDev, *mBoundZ0Dev, *mBoundZMaxDev;


    dim3 mCuBlockSize;
    dim3 mCuVolGridSize;
    dim3 mCuSurfBlockSize;
    dim3 mCuSurfGridSizeZ;
    dim3 mCuSurfGridSizeY;
    dim3 mCuSurfGridSizeX;

    //The following arrays store the same data as in Interconnect **mInterconnects but in a convenient way for kernel calls
    int* mIcMshiftDev;
    int* mIcNshiftDev;
    int* mIcNdimDev;
    float** mIcOwnerTemperatureDev;
    float** mIcOwnerConductivityDev;
    float* mIcSourceHDev;

};



#endif /* BLOCK3DCUDA_H_ */
