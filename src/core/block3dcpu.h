/*
 * block3dcpu.h
 *
 *  Created on: Nov 5, 2014
 *      Author: dglyzin
 */

#ifndef BLOCK3DCPU_H_
#define BLOCK3DCPU_H_

#include "block3d.h"
#include "mystdint.h"


class Block3dCpu: public Block3d {
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
    float GetBorderBoundFlow(uint16_t bound, int volumeIdx, float area, float size, char direction);
    float GetBorderIcFlow(uint16_t icIdx, int sideMidx, int sideNidx, int volumeIdx, float area, float size, char direction);
    void  ComputeHeatFlow();

};



#endif /* BLOCK3DCPU_H_ */
