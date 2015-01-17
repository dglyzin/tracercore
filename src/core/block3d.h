/*
 * block3d.h
 *
 *  Created on: Oct 9, 2014
 *      Author: dglyzin
 */

#ifndef BLOCK3D_H_
#define BLOCK3D_H_

#include "mystdint.h"
#include "interconnect.h"
#include <fstream>
//#include "domain3d.h"

///Virtual building block for 3d domain
/* Block3d is used as an ancestor to every architecture-specific block
 * The workflow is as follows
 * 2. call SetData to fill everything the block needs to know
 * 3. PrepareEngine is an architecture-specific arranging of data arrays
 * 4. call ProcessOneStep in a loop (architecture-specific)
 * 6. GetData to collect info from multiple blocks
 * 7. call CleanData to release allocated resources
 */
class Block3d {

public:
    virtual ~Block3d() {};

    //Do all preparations after load (including copy to CUDA memory for cuda block)
    virtual void PrepareEngine(float timeStep) = 0;
    virtual void SetHeatSourceData(int heatSourceCount, float *heatSourceDensity) = 0;
    virtual void SetSoilData(int soilCount, float* frozenSoilConductivity, float* thawedSoilConductivity,
                   float* drySoilSpecHeat, float* frozenSoilVolHeat, float* thawedSoilVolHeat,
                   float* drySoilDensity, float* waterPerDrySoilMass, float* phaseTransitionPoint,
                   int* liquidWaterGraphLength, float* liquidWaterGraphs,
                   float* temperatureGraph, float* hMelt, float* liqPartAtTr,
                   float waterLatentHeat) = 0;
    virtual void SetCavernData(int cavernCount, int* cavernBound) = 0;
    virtual void SetBoundData(int boundCount, int *boundTypes, float *temperatureLag,
                   float *alphaConvection, float *boundValues) = 0;

    virtual void FillInterconnectBuffer(int idx) = 0;

    virtual void ProcessOneStep(int month) = 0;
    virtual void StoreData() = 0;
    virtual void ReleaseResources() = 0;
            void ReleaseBasicResources();


    void LoadFromFile(std::ifstream &,std::ifstream &, int blockNum, int blockLocationNode, int blockLocationDevice);
    void SaveBlockDataBin(std::string binFileName);
    void SaveBlockDataTxt(std::string txtFileName);
    //this function is here as there is no need to run it on gpu - it is used only once
    //arguments are passed here again because in setsoildata gpu block will copy data to gpu directly
    void ComputeEnthalpy(float* liqPartAtTr, float** liqidWaterGraphPtrs, int* liquidWaterGraphLength, float* hMelt,
                float* thawedSoilVolHeat, float* frozenSoilVolHeat, float* drySoilDensity, float* waterPerDrySoilMass,
                float* phaseTransitionPoint);

    float GetMinH();
    int GetLocationNode(){return mLocationNode;};
    int GetLocationDevice(){return mLocationDevice;};
    void SetInterconnectArray(Interconnect **icArray, int count){mInterconnects = icArray; mInterconnectCount = count;};
    int GetSideMsize(int);
    int GetSideNsize(int);
    int GetVolumeCount(){return mXcount*mYcount*mZcount;};

protected:
    void LoadGeometryFromFile(std::ifstream &input_file);

    // ********COMPUTATIONAL RESOURCES****
    int mSelfIdx;
    int mLocationNode;
    int mLocationDevice;

    // ************BLOCK STRUCTURE******************
    int mXoffset;
    int mYoffset;
    int mZoffset;
    //number of volumes along axes
    int mXcount;
    int mYcount;
    int mZcount;
    int mZSliceSize;
    float* mSizes[3];

    // **********STATE******************
    //Temperature
    float* mTemperature;
    //Volumetric Enthalpy
    float* mEnthalpy;
    //Whether the block is at phase transition temperature
    char* mAtPhaseTransition;
    //liquid water percentage in the block
    float* mLiquidPart;

    int mMonth;
    float mTimeStep;

    // **********STATE HELPERS************
    float* mRecConductivity;
    //heat flow between every pair of blocks, contains +1 element in every dimension
    int    mFlowSize[3];
    float* mHeatFlow[3];

    // ************Properties***************
    uint8_t *mBodyType; //soil(0) or cavern(1)
    uint16_t * mBody;   //number of soil or cavern
    //heat source type: 0 for none, index+1 for others
    uint16_t *mHeatSources;
    uint8_t  *mBorderType[6]; //bound (0) or interconnect(1)
    uint16_t *mBorder[6];
    int mBorderSizeM[3]; //major
    int mBorderSizeN[3]; //minor
    int mBorderSizeR[3]; //the rest
    int mBorderStrideM[3];
    int mBorderStrideN[3];
    int mBorderStrideR[3];
    int mBodyStrideM[3];
    int mBodyStrideN[3];
    int mBodyStrideR[3];
    int mBorderSideRidx[6]; //index along the rest axis for this border (0 or max)

    // ****data for soils, bounds, ic and sources***********
    // ************ soil parameters*******************************
    int mSoilCount;
    float* mFrozenSoilConductivity;
    float* mThawedSoilConductivity;
    float* mDrySoilSpecHeat;
    float* mFrozenSoilVolHeat;
    float* mThawedSoilVolHeat;
    float* mDrySoilDensity;
    float* mWaterPerDrySoilMass;
    float* mPhaseTransitionPoint;
    int* mLiquidWaterGraphLength;
    float* mLiquidWaterGraphs;
    //the next four arrays are computed in this core (by Domain3d)
    float** mLiqidWaterGraphPtrs;
    float* mTemperatureGraph;
    float* mHMelt;       //Enthalpy for 100% liquid water
    float* mLiqPartAtTr; //\bar w(u^*), minimum of liquid water at phase transition point

    float mWaterLatentHeat;

    //caverns
    int mCavernCount;
    int* mCavernBound;

    //pointer to interconnect array
    Interconnect **mInterconnects;
    int mInterconnectCount;

    //******* bound parameters
    int mBoundCount;
    int *mBoundTypes;
    float *mTemperatureLag;
    float *mAlphaConvection;
    float *mBoundValues;

    // ***************heat source  parameters***
    int mHeatSourceCount;
    float *mHeatSourceDensity;

};

#endif /* BLOCK3D_H_ */
