/*
 * domain3d.h
 *
 *  Created on: Oct 16, 2014
 *      Author: dglyzin
 */

#ifndef DOMAIN3D_H_
#define DOMAIN3D_H_


#include "mystdint.h"
#include "block3d.h"
#include "interconnect.h"
#include <cstdio>
#include <string>

///Domain3d manages IO and processing Blocks3d and their Interconnects
/*
 * Core units: day, KJoule, meter, Kelvin, kg
 * Start time is stored in binary file and is made the current time of the system
 * Computation time is loaded from binary file as well by default but can be changed from command line
 *
 */
class Domain3d{
public:
    //here is the data flow: load, prepare, process one step many times, release resources.
    //Anytime we can save the data to binary or text file
    EXPORT_XX int LoadFromBinaryFile(std::string filename, int locationNode);

    EXPORT_XX void ProcessOneStep();

    EXPORT_XX void SaveToFile(std::string filename, int saveText);

    EXPORT_XX void ReleaseResources();


    EXPORT_XX float GetTime(){return mTime;}

    EXPORT_XX float GetFinishTime(void);

    EXPORT_XX int GetVolumeCount(void);
    EXPORT_XX int GetStepCount(void);
    //EXPORT_XX int GetMonth(){};
 /*

    EXPORT_XX uint8_t GetMonth(void);
    EXPORT_XX float GetTime(void);
    EXPORT_XX float GetCompTime(void);


*/
private:
    int  ReadBinaryData(std::string namebase);
    void PrepareData();
    void StoreData(); //make every block data available on hosts
    void RecvAndSaveBlockData(std::string binFileName, std::string textFileName, int saveText);
    void SendBlockDataToRoot();

    void ComputeMonth();
    void GenerateTemperatureDependence();
    void ComputeSoilProperties();
    float ComputeTimeStep();

    void ProcessInterconnects();

    void SaveDomainInfoBin(std::string binFileName);
    void SaveDomainInfoTxt(std::string txtFileName);

    //void SaveStaticDataBin(std::string binFileName);
    //void SaveStaticDataTxt(std::string txtFileName);

    int mLocationNode;
    uint8_t mGeneratedInput;
    int mMonth;
    //number of steps from the start
    int mStepCounter;


    /************GROUP 0 - DOMAIN STRUCTURE******************/
    float mTime;
    float mFinishTime;
    //time step, computed or loaded, whatever is smaller
    float mTimeStep;
    float mSavesPerMonth;
    //constants actually
    float mIceSpecHeat;
    float mWaterSpecHeat;
    float mWaterLatentHeat;
    float mIceVolumetricExpansion;

    //total number of blocks
    int mBlockCount;
    Block3d** mBlocks;

    // ************GROUP 2 - Arrays with soil parameters*******************************
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

    //the next four arrays are computed in this core
    float** mLiqidWaterGraphPtrs;
    float* mTemperatureGraph;
    float* mHMelt;       //Enthalpy for 100% liquid water
    float* mLiqPartAtTr; //\bar w(u^*), minimum of liquid water at phase transition point

    //Caverns
    int mCavernCount;
    int *mCavernBound;

    // ************GROUP 3 - Arrays with boundary parameters*******************************
    int mBoundCount;
    int *mBoundTypes;
    float *mTemperatureLag;
    float *mAlphaConvection;
    float *mBoundValues;

    // ************ Interconnects*******************************
    //arrays from file
    int mInterconnectCount;
    Interconnect ** mInterconnects;

    // ************GROUP 4 - Arrays with heat sources parameters*******************************
    int mHeatSourceCount;
    float *mHeatSourceDensity;
};

#endif
