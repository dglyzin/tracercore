#include <iostream>
#include <fstream>
#include <cstring>
#include "domain3d.h"
#include <string>
#include <omp.h>
#include <sstream>
#include <iomanip>
#include <stdlib.h>
#include "block3dcpu.h"
#include "block3dcuda.h"
#include "block3d.h"
#include "approx.h"


EXPORT_XX
int Domain3d::LoadFromBinaryFile(std::string filename, int locationNode){
    int status;
    mLocationNode = locationNode;
    status = ReadBinaryData(filename);
    if (status) return status;
    //prepare blocks data and make additional convenience precomputations
    PrepareData();
    return 0;
}




int Domain3d::ReadBinaryData(std::string filename){
    std::cout<<"loading data from "<<filename<<std::endl;
    std::ifstream input_file_dom ((filename+std::string(".dom")).c_str(), std::ios::in | std::ios::binary);
    std::ifstream input_file_bin ((filename+std::string(".bin")).c_str(), std::ios::in | std::ios::binary);

    uint8_t version1, version2, v1,v2;
    //Read state file header
    input_file_bin.read ((char*)&version1, 1);
    if (version1!=253){
      std::cout<<"Wrong .bin input file format"<<std::endl;
      return 1;
    }

    input_file_bin.read ((char*)&v1, 1);
    input_file_bin.read ((char*)&v2, 1);
    std::cout<<"Reading state file format v"<<(int)v1<<"."<<(int)v2 <<std::endl;
    if (v1 > 2){
      std::cout<<"Input file format is newer than supported "<<std::endl;
      return 1;
    }

    input_file_bin.read((char*)&mGeneratedInput, 1);
    if (mGeneratedInput)
      std::cout<<"File is script-generated."<<std::endl;
    else
      std::cout<<"File contains computed enthalpy."<<std::endl;

    input_file_bin.read ((char*)&mTime, sizeof(float));
    std::cout<<"Current time is "<< mTime << std::endl;

    //Read domain file header
    //1. Read version info
    input_file_dom.read ((char*)&version1, 1);
    if (version1!=254){
        std::cout<<"Wrong .dom input file format"<<std::endl;
        return 1;
    }

    input_file_dom.read ((char*)&version1, 1);
    input_file_dom.read ((char*)&version2, 1);
    std::cout<<"Reading domain file format v"<<(int)version1<<"."<<(int)version2 <<std::endl;
    if (version1 > 2){
        std::cout<<"Input file format is newer than supported "<<std::endl;
        return 1;
    }
    if ((v1!=version1)||(v2!=version2)){
        std::cout<<"Domain and state files version do not match! "<<std::endl;
        return 1;
    }


    //2. Read parameters
    input_file_dom.read ((char*)&mFinishTime, sizeof(float));
    input_file_dom.read ((char*)&mTimeStep, sizeof(float));
    input_file_dom.read ((char*)&mSavesPerMonth, sizeof(float));
    input_file_dom.read ((char*)&mIceSpecHeat, sizeof(float));
    input_file_dom.read ((char*)&mWaterSpecHeat, sizeof(float));
    input_file_dom.read ((char*)&mWaterLatentHeat, sizeof(float));// = 335.f; // kJ/kg  (kappa)
    input_file_dom.read ((char*)&mIceVolumetricExpansion, sizeof(float));// = 1.1; //dimensionless, nu

    //3. Read global grid
    int gxc,gyc,gzc;
    float* tmp_storage;
    input_file_dom.read ((char*)&gxc, sizeof(int));
    input_file_dom.read ((char*)&gyc, sizeof(int));
    input_file_dom.read ((char*)&gzc, sizeof(int));
    tmp_storage = (float*) malloc(gxc*sizeof(float));
    input_file_dom.read ((char*)tmp_storage, gxc*sizeof(float));
    free(tmp_storage);
    tmp_storage = (float*) malloc(gyc*sizeof(float));
    input_file_dom.read ((char*)tmp_storage, gyc*sizeof(float));
    free(tmp_storage);
    tmp_storage = (float*) malloc(gzc*sizeof(float));
    input_file_dom.read ((char*)tmp_storage, gzc*sizeof(float));
    free(tmp_storage);


    //3. Read block structure
    input_file_dom.read ((char*)&mBlockCount, sizeof(int));

    mBlocks = new Block3d* [mBlockCount];

    for (int blockIdx=0; blockIdx<mBlockCount; blockIdx++){
        int blockLocationNode;
        int blockLocationDevice;
        input_file_dom.read ((char*)&blockLocationNode, sizeof(int));
        input_file_dom.read ((char*)&blockLocationDevice, sizeof(int));
        std::cout<<"Block " << blockIdx << " goes to node #"<< blockLocationNode << " device #"<< blockLocationDevice <<std::endl;
        if (blockLocationNode == mLocationNode){//local block
            if (blockLocationDevice>0)
                mBlocks[blockIdx] = new Block3dCuda;
            else
                mBlocks[blockIdx] = new Block3dCpu;
        }
        else {//foreign block
            mBlocks[blockIdx] = new Block3dCpu;//Block3dStub;
        }
        mBlocks[blockIdx]->LoadFromFile(input_file_dom, input_file_bin, blockIdx, blockLocationNode, blockLocationDevice);
    }


    //Soil data
    input_file_dom.read((char*)&mSoilCount,sizeof(int));
    std::cout<<"Loading "<<mSoilCount<<" types of soil"<<std::endl;

    mFrozenSoilConductivity=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mFrozenSoilConductivity,mSoilCount*sizeof(float));
    mThawedSoilConductivity=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mThawedSoilConductivity,mSoilCount*sizeof(float));
    mDrySoilSpecHeat=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mDrySoilSpecHeat,mSoilCount*sizeof(float));
    mFrozenSoilVolHeat=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mFrozenSoilVolHeat,mSoilCount*sizeof(float));
    mThawedSoilVolHeat=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mThawedSoilVolHeat,mSoilCount*sizeof(float));
    mDrySoilDensity=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mDrySoilDensity,mSoilCount*sizeof(float));
    mWaterPerDrySoilMass=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mWaterPerDrySoilMass,mSoilCount*sizeof(float));
    mPhaseTransitionPoint=(float*) malloc(mSoilCount*sizeof(float));
    input_file_dom.read((char*)mPhaseTransitionPoint,mSoilCount*sizeof(float));
    mLiquidWaterGraphLength=(int*) malloc(mSoilCount*sizeof(int));
    input_file_dom.read((char*)mLiquidWaterGraphLength,mSoilCount*sizeof(int));
    int totalgrapharraysize=0;
    for (int idx = 0; idx<mSoilCount;idx++)
        totalgrapharraysize+= mLiquidWaterGraphLength[idx];
    totalgrapharraysize*=2;
    mLiquidWaterGraphs = (float*) malloc(totalgrapharraysize*sizeof(float));
    input_file_dom.read((char*)mLiquidWaterGraphs,totalgrapharraysize*sizeof(float));

    //cavern data
    input_file_dom.read((char*)&mCavernCount,sizeof(int));
    std::cout<<"Loading "<<mCavernCount<<" types of cavern"<<std::endl;
    //every cavern has 6 links to bounds
    mCavernBound = (int*) malloc( 6 * mCavernCount*sizeof(int));
    input_file_dom.read((char*)mCavernBound, 6 * mCavernCount*sizeof(int));



    //Boundary conditions data
    input_file_dom.read((char*)&mBoundCount,sizeof(int));
    std::cout<<"Loading "<<mBoundCount<<" types of boundaries (including default zero flow)"<<std::endl;

    mBoundTypes = (int*) malloc(mBoundCount*sizeof(int));
    input_file_dom.read((char*)mBoundTypes, mBoundCount*sizeof(int));

    mTemperatureLag = (float*) malloc(12*mBoundCount*sizeof(float));
    input_file_dom.read((char*)mTemperatureLag, 12*mBoundCount*sizeof(float));
    mAlphaConvection = (float*) malloc(12*mBoundCount*sizeof(float));
    input_file_dom.read((char*)mAlphaConvection, 12*mBoundCount*sizeof(float));
    mBoundValues = (float*) malloc(12*mBoundCount*sizeof(float));
    input_file_dom.read((char*)mBoundValues, 12*mBoundCount*sizeof(float));

    //interconnects
    input_file_dom.read((char*)&mInterconnectCount,sizeof(int));
    std::cout<<"Loading "<<mInterconnectCount<<" types of interconnects"<<std::endl;

    mInterconnects = new Interconnect* [mInterconnectCount];
    for (int icIdx=0; icIdx<mInterconnectCount; icIdx++){
        int owner, source, sourceSide, mshift, nshift;
        float sourceH;
        input_file_dom.read((char*)&owner, sizeof(int));
        input_file_dom.read((char*)&source, sizeof(int));
        input_file_dom.read((char*)&sourceSide, sizeof(int));
        input_file_dom.read((char*)&sourceH, sizeof(float));
        input_file_dom.read((char*)&mshift, sizeof(int));
        input_file_dom.read((char*)&nshift, sizeof(int));
        mInterconnects[icIdx] = new Interconnect(owner, source, sourceSide, sourceH, mshift, nshift,
                                                    mBlocks[owner]->GetLocationNode(), mBlocks[owner]->GetLocationDevice(),
                                                    mBlocks[source]->GetLocationNode(), mBlocks[source]->GetLocationDevice(),
                                                    mBlocks[source]->GetSideMsize( sourceSide), mBlocks[source]->GetSideNsize( sourceSide));
        mInterconnects[icIdx]->PrepareData(mLocationNode);
        std::cout<<"ic#"<<icIdx<<" owner="<<owner<<" source="<< source<<", source side="<<sourceSide<< std::endl;
    }
    //set interconnect array pointer in every block
    for (int blockIdx=0; blockIdx<mBlockCount; blockIdx++)
        mBlocks[blockIdx]->SetInterconnectArray(mInterconnects, mInterconnectCount);


    //Sources data
    input_file_dom.read((char*)&mHeatSourceCount,sizeof(int));
    std::cout<<"Loading "<<mHeatSourceCount<<" types of heat sources (including one empty)"<<std::endl;
    mHeatSourceDensity = (float*) malloc(12*mHeatSourceCount*sizeof(float));
    input_file_dom.read((char*)mHeatSourceDensity, 12*mHeatSourceCount*sizeof(float));

    input_file_dom.close();
    input_file_bin.close();
    return 0;
}


// *************  INIT ROUTINES ***************
void Domain3d::PrepareData(){
    //Generate once all data needed for every block and for domain itself
    ComputeMonth();
    //create convenience pointers
    mLiqidWaterGraphPtrs = (float**) malloc(sizeof(float*)*mSoilCount);
        int shift=0;
        for (int idx = 0; idx<mSoilCount;idx++){
            mLiqidWaterGraphPtrs[idx] = mLiquidWaterGraphs + shift;
            shift+= 2*mLiquidWaterGraphLength[idx];
        }

    //compute soil properties
    mHMelt = (float*) malloc(mSoilCount*sizeof(float));
    mLiqPartAtTr = (float*) malloc(mSoilCount*sizeof(float));
    ComputeSoilProperties();
    //and create a temperature-of-enthalpy dependence
    mTemperatureGraph = (float*) malloc(sizeof(float)*TEMP_GRAPH_LEN*2*mSoilCount);
    GenerateTemperatureDependence();



    float ts = ComputeTimeStep();
    std::cout << "Computed time step: "<< ts << ", loaded time step = "<< mTimeStep <<std::endl;
    if (ts < mTimeStep){
        mTimeStep = ts;
        std::cout << "Computed time step selected"<<std::endl;
    } else
        std::cout << "Loaded time step selected"<<std::endl;

    mStepCounter = 0;

    //send prepared data and data loaded from file (soil, bound, ic, source)
    //stubs will not do anything (they are located on different nodes)
    //cpublocks will copy data
    //gpublocks copy directly into gpu memory
    for (int idx = 0; idx<mBlockCount; idx++){
        std::cout << "Setting block "<< idx << " properties"<<std::endl;
        std::cout << "1. Soils"<<std::endl;
        mBlocks[idx]->SetSoilData(mSoilCount, mFrozenSoilConductivity, mThawedSoilConductivity,
                                  mDrySoilSpecHeat, mFrozenSoilVolHeat, mThawedSoilVolHeat,
                                  mDrySoilDensity, mWaterPerDrySoilMass, mPhaseTransitionPoint,
                                  mLiquidWaterGraphLength, mLiquidWaterGraphs,
                                  mTemperatureGraph, mHMelt, mLiqPartAtTr, mWaterLatentHeat);
        std::cout << "2. Caverns"<<std::endl;
        mBlocks[idx]->SetCavernData(mCavernCount, mCavernBound);
        std::cout << "3. Bounds"<<std::endl;
        mBlocks[idx]->SetBoundData(mBoundCount, mBoundTypes, mTemperatureLag,
                                   mAlphaConvection, mBoundValues);
        std::cout << "4. Heat Sources"<<std::endl;
        mBlocks[idx]->SetHeatSourceData(mHeatSourceCount, mHeatSourceDensity);
        if (mGeneratedInput)
            std::cout << "Computing enthalpy"<<std::endl;
            mBlocks[idx]->ComputeEnthalpy(mLiqPartAtTr, mLiqidWaterGraphPtrs, mLiquidWaterGraphLength, mHMelt,
                    mThawedSoilVolHeat, mFrozenSoilVolHeat, mDrySoilDensity, mWaterPerDrySoilMass, mPhaseTransitionPoint);
        std::cout << "Preparing engine"<<std::endl;
        mBlocks[idx]->PrepareEngine(mTimeStep); //should be called only once for every block
    }
    mGeneratedInput = 0;
}


//jan feb mar apr may jun jul aug sep oct nov dec
//31  28  31  30  31  30  31  31  30  31  30  31
//0-31 59 90  120 151  181 212 243 273 304 334 365
void Domain3d::ComputeMonth(){

    float dayofyear = mTime - float(int(mTime/365.0)*365);
    if (dayofyear<31.0f)
      mMonth = 0;
    else if ((dayofyear>=31.0f) && (dayofyear<59.0f))
      mMonth = 1;
    else if ((dayofyear>=59.0f) && (dayofyear<90.0f))
      mMonth = 2;
    else if ((dayofyear>=90.0f) && (dayofyear<120.0f))
      mMonth = 3;
    else if ((dayofyear>=120.0f) && (dayofyear<151.0f))
      mMonth = 4;
    else if ((dayofyear>=151.0f) && (dayofyear<181.0f))
      mMonth = 5;
    else if ((dayofyear>=181.0f) && (dayofyear<212.0f))
      mMonth = 6;
    else if ((dayofyear>=212.0f) && (dayofyear<243.0f))
      mMonth = 7;
    else if ((dayofyear>=243.0f) && (dayofyear<273.0f))
      mMonth = 8;
    else if ((dayofyear>=273.0f) && (dayofyear<304.0f))
      mMonth = 9;
    else if ((dayofyear>=304.0f) && (dayofyear<334.0f))
      mMonth = 10;
    else
      mMonth = 11;

}

void Domain3d::GenerateTemperatureDependence(){
    /* enthalpy and liquid water dependencies are stored in memory in decreasing order
    *
    * we will start from mPhaseTransitionPoint (this is the temperature where enthalpy=0 and possibly >0
    * we will use this dependence to compute temperature from enthalpy only if enthalpy < 0
    *
    * the next possible point is where water graph implies 100% liquid water by extrapolation
    * if this point is > mPhaseTransitionPoint then it is skipped
    * (to epmhasize the segment where enthalpy is decreasing only due to lowering temperature and not to phase transition)
    *
    * next we add all the points from liquid water graph
    *
    * finally we divide the segment [-TEMP_GRAPH_LEN, lowest liqid water graph point] into equal parts
    * to fill the remaining points in the Dependence
    *
    */
   float temperature, FrCapacity, LiqPart, LiqPartAtTr, enthalpy;
   for (int soilidx=0; soilidx<mSoilCount; soilidx++){
       std::cout<<"printing dependence for soil "<<soilidx<<std::endl;
       //TODO check if this is ok to use FrCapacity
       FrCapacity = mFrozenSoilVolHeat[soilidx];
       LiqPartAtTr =mLiqPartAtTr[soilidx];
       float* pDependence = mLiqidWaterGraphPtrs[soilidx];
       int deplength = mLiquidWaterGraphLength[soilidx];
       //first point
       int idx = 0;
       temperature = mPhaseTransitionPoint[soilidx];
       enthalpy = 0;
       mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+0] = enthalpy;
       mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+1] = temperature;
       std::cout <<temperature << ":"<< enthalpy<<":"<<GetLiquidPartForSoilTemp(pDependence, deplength, temperature)<<std::endl;
       //second point, if necessary
       idx = 1;


       float extrapolated100 = (1.0f-pDependence[2+1])*(pDependence[0]-pDependence[2])/(pDependence[1]-pDependence[2+1]) + pDependence[2];
       if (extrapolated100 < temperature){
           temperature = extrapolated100;
           enthalpy = FrCapacity*(temperature-mPhaseTransitionPoint[soilidx]);
           mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+0] = enthalpy;
           mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+1] = temperature;
           std::cout <<temperature << ":"<< enthalpy<<":"<<GetLiquidPartForSoilTemp(pDependence, deplength, temperature)<<std::endl;
           idx = 2;
       }
       //liquid water graph points
       for (int point = 0;point<deplength;point++){
           if (idx<TEMP_GRAPH_LEN){
               temperature = pDependence[2*point];
               LiqPart = GetLiquidPartForSoilTemp(pDependence, deplength, temperature);
               enthalpy = FrCapacity*(temperature-mPhaseTransitionPoint[soilidx])+mWaterLatentHeat*mDrySoilDensity[soilidx]*mWaterPerDrySoilMass[soilidx]*(LiqPart-LiqPartAtTr);
               mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+0] = enthalpy;
               mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+1] = temperature;
               std::cout <<temperature << ":"<< enthalpy<<":"<<GetLiquidPartForSoilTemp(pDependence, deplength, temperature)<<std::endl;
               idx++;
           }
       }
       //and the rest
       float lasttemp = temperature;
       int lastidx = idx-1;
       for  (; idx<TEMP_GRAPH_LEN; idx++){
           temperature = lasttemp-float(idx-lastidx);
           LiqPart = GetLiquidPartForSoilTemp(pDependence, deplength, temperature);
           enthalpy = FrCapacity*(temperature-mPhaseTransitionPoint[soilidx])+mWaterLatentHeat*mDrySoilDensity[soilidx]*mWaterPerDrySoilMass[soilidx]*(LiqPart-LiqPartAtTr);
           mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+0] = enthalpy;
           mTemperatureGraph[soilidx*TEMP_GRAPH_LEN*2+idx*2+1] = temperature;
           std::cout <<temperature << ":"<< enthalpy<<":"<<GetLiquidPartForSoilTemp(pDependence, deplength, temperature)<<std::endl;
       }
   }

}


void Domain3d::ComputeSoilProperties(){
    for (int soilidx=0; soilidx<mSoilCount; soilidx++){
        mLiqPartAtTr[soilidx] = GetLiquidPartForSoilTemp(mLiqidWaterGraphPtrs[soilidx], mLiquidWaterGraphLength[soilidx],mPhaseTransitionPoint[soilidx]);
        mHMelt[soilidx] = mWaterLatentHeat*mDrySoilDensity[soilidx]*mWaterPerDrySoilMass[soilidx] * (1- mLiqPartAtTr[soilidx]);
        //DrySoilSpecHeatDev was computed by input_generator.py
    }
}



float Domain3d::ComputeTimeStep(){
    float minh, minc, maxl;

    minc = mFrozenSoilVolHeat[0];
    maxl = mFrozenSoilConductivity[0];
    for(int i = 1; i< mSoilCount; i++){
        if (minc > mFrozenSoilVolHeat[i])
            minc = mFrozenSoilVolHeat[i];
        if (maxl < mFrozenSoilConductivity[i])
            maxl = mFrozenSoilConductivity[i];
    }
    minh =  mBlocks[0]->GetMinH();
    for (int idx = 1; idx<mBlockCount; idx++){
        float h =  mBlocks[idx]->GetMinH();
        if (h<minh)
            minh = h;
    }
    return minh*minh*minc /(6.f * maxl);
}


// *********** ITERATIONS *************
EXPORT_XX
void Domain3d::ProcessOneStep(){
    //at this time we know current temperature, conductivity and enthalpy
    ProcessInterconnects();
    for (int idx=0; idx<mBlockCount; idx++)
        mBlocks[idx]->ProcessOneStep(mMonth);

    mTime += mTimeStep;
    ComputeMonth();
    mStepCounter ++;

}


void Domain3d::ProcessInterconnects(){
    //we want to process every interconnect
    //interconnects are one-sided
    //if source or owner block is not on this node, it does nothing
    for(int idx=0; idx<mInterconnectCount; idx++){
        int source = mInterconnects[idx]->mSource;
        mBlocks[source]->FillInterconnectBuffer(idx);
        mInterconnects[idx]->SendRecv(mLocationNode);
    }

}

// ****** SAVING ROUTINES *************
EXPORT_XX
void Domain3d::SaveToFile(std::string namebase, int saveText){
    //every node copies block data to cpu
    StoreData();
    if (mLocationNode == ROOT_NODE){
        std::ostringstream o;
        o << std::fixed << std::setw( 7 ) << std::setprecision( 1 )
        << std::setfill( '0' ) << GetTime();
        std::string fbinname = namebase+std::string("_")+o.str();
        std::string ftxtname;
        if (saveText) {
            ftxtname = namebase+std::string("_")+o.str()+std::string(".txt");
            //save general domain settings
            SaveDomainInfoBin(fbinname);
            SaveDomainInfoTxt(ftxtname);
            //save block info
            RecvAndSaveBlockData(fbinname, ftxtname, saveText);
            //save soils etc
            //SaveStaticDataBin(fbinname);
            //SaveStaticDataTxt(ftxtname);
        }
        else{
            ftxtname = std::string("");
            //save general domain settings
            SaveDomainInfoBin(fbinname);
            //save block info
            RecvAndSaveBlockData(fbinname, ftxtname, saveText);
            //save soils etc
           // SaveStaticDataBin(fbinname);
        }
    }
    else
        SendBlockDataToRoot();
}

void Domain3d::StoreData(){
    for (int idx = 0; idx<mBlockCount; idx++)
        mBlocks[idx]->StoreData();
}

void Domain3d::SaveDomainInfoBin(std::string fileNameBase){
    //std::cout<<"Saving data to "<<filename<<std::endl<<std::endl;
    std::ofstream dom_file((fileNameBase+std::string(".dom")).c_str(), std::ios::out | std::ios::binary);
    uint8_t version1=254;
    uint8_t version2=BINARY_FILE_VERSION_MAJ;
    uint8_t version3=BINARY_FILE_VERSION_MIN;
    dom_file.write((char*)&version1, 1);
    dom_file.write((char*)&version2, 1);
    dom_file.write((char*)&version3, 1);
    dom_file.write((char*)&mFinishTime, sizeof(float));
    dom_file.write((char*)&mTimeStep, sizeof(float));
    dom_file.write((char*)&mSavesPerMonth, sizeof(float));
    dom_file.write((char*)&mIceSpecHeat, sizeof(float));
    dom_file.write((char*)&mWaterSpecHeat, sizeof(float));
    dom_file.write((char*)&mWaterLatentHeat, sizeof(float));
    dom_file.write((char*)&mIceVolumetricExpansion, sizeof(float));
    dom_file.write((char*)&mBlockCount, sizeof(int));
    dom_file.close();

    std::ofstream bin_file ((fileNameBase+std::string(".bin")).c_str(), std::ios::out | std::ios::binary);
    bin_file.write((char*)&mGeneratedInput,1);
    bin_file.write((char*)&mTime, sizeof(float));
    bin_file.close();
}

void Domain3d::SaveDomainInfoTxt(std::string txtFileName){
    //std::cout<<"Saving results to "<<filename<<std::endl<<std::endl;
    std::ofstream res_file(txtFileName.c_str(), std::ios::out);

    res_file<<"Time:"<<mTime<<std::endl;
    res_file<<"Finish Time:"<<mFinishTime<<std::endl;
    res_file<<"Time step:"<<mTimeStep<<std::endl;
    res_file<<"Saves per month:"<<mSavesPerMonth<<std::endl;

    res_file<<"Bounds types: ";
    for (int idx =0;idx<mBoundCount; idx++)
        res_file<< mBoundTypes[idx]<<" ";
    res_file<<std::endl;

    res_file<<"Temp Lag: ";
    for (int idx =0;idx<mBoundCount; idx++)
        res_file<< mTemperatureLag[idx]<<" ";
    res_file<<std::endl;

    res_file<<"Alpha: ";
    for (int idx =0;idx<mBoundCount; idx++)
        res_file<< mAlphaConvection[idx]<<" ";
    res_file<<std::endl;


    res_file<<"Bound Value: ";
    for (int idx =0;idx<mBoundCount; idx++)
        res_file<< mBoundValues[idx]<<" ";
    res_file<<std::endl;




       /*       //interconnects
       input_file.read((char*)&mInterconnectCount,sizeof(int));
       std::cout<<"Loading "<<mInterconnectCount<<" types of interconnects"<<std::endl;
       mInterconnectOwners = (int*) malloc(mInterconnectCount*sizeof(int));
       input_file.read((char*)mInterconnectOwners, mInterconnectCount*sizeof(int));
       mInterconnectSources = (int*) malloc(mInterconnectCount*sizeof(int));
       input_file.read((char*)mInterconnectSources, mInterconnectCount*sizeof(int));
       mInterconnectSourceSide = (int*) malloc(mInterconnectCount*sizeof(int));
       input_file.read((char*)mInterconnectSourceSide, mInterconnectCount*sizeof(int));
       mInterconnectSourceH = (float*) malloc(mInterconnectCount*sizeof(float));
       input_file.read((char*)mInterconnectSourceH, mInterconnectCount*sizeof(float));
       mInterconnectMshift = (int*) malloc(mInterconnectCount*sizeof(int));
       input_file.read((char*)mInterconnectMshift, mInterconnectCount*sizeof(int));
       mInterconnectNshift = (int*) malloc(mInterconnectCount*sizeof(int));
       input_file.read((char*)mInterconnectNshift, mInterconnectCount*sizeof(int));
       for (int idx = 0; idx< mInterconnectCount; idx++)
           std::cout<<"side of ic#"<<idx<<" is "<<mInterconnectSourceSide[idx]<<std::endl;
*/
    res_file<<"Number of blocks:"<<mBlockCount<<std::endl;
}


void Domain3d::RecvAndSaveBlockData(std::string fileNameBase, std::string textFileName, int saveText){
    //collect blocks one by one from their respective owners
    for (int idx = 0; idx<mBlockCount; idx++)
        if (mBlocks[idx]-> GetLocationNode() != ROOT_NODE){
            //TODO ALLOC-RECEIVE-SAVE-FREE
            //SaveBlockDataBin(std::string binFileName, idx);
            //if (savetext)
            //    SaveBlockDataTxt(std::string textFileName, idx);
        }
        else{
            //just save
            mBlocks[idx]-> SaveBlockDataBin(fileNameBase);
            if (saveText)
                mBlocks[idx]-> SaveBlockDataTxt(textFileName);
        }
}



void Domain3d::SendBlockDataToRoot(){
    //send every local block info to the root node
    for (int idx = 0; idx<mBlockCount; idx++)
        if (mBlocks[idx]-> GetLocationNode() == mLocationNode){
            //TODO SEND mBlocks[idx]->everything
        }
}


/*
void Domain3d::SaveStaticDataBin(std::string fileNameBase){
    std::ofstream dom_file((fileNameBase+std::string(".dom")).c_str(), std::ios::out | std::ofstream::app | std::ios::binary);
    //Soil data
    dom_file.write((char*)&mSoilCount,sizeof(int));
    dom_file.write((char*)mFrozenSoilConductivity,mSoilCount*sizeof(float));
    dom_file.write((char*)mThawedSoilConductivity,mSoilCount*sizeof(float));
    dom_file.write((char*)mDrySoilSpecHeat,mSoilCount*sizeof(float));
    dom_file.write((char*)mFrozenSoilVolHeat,mSoilCount*sizeof(float));
    dom_file.write((char*)mThawedSoilVolHeat,mSoilCount*sizeof(float));
    dom_file.write((char*)mDrySoilDensity,mSoilCount*sizeof(float));
    dom_file.write((char*)mWaterPerDrySoilMass,mSoilCount*sizeof(float));
    dom_file.write((char*)mPhaseTransitionPoint,mSoilCount*sizeof(float));
    dom_file.write((char*)mLiquidWaterGraphLength,mSoilCount*sizeof(int));
    int totalgrapharraysize=0;
    for (int idx = 0; idx<mSoilCount;idx++)
        totalgrapharraysize+= mLiquidWaterGraphLength[idx];
    totalgrapharraysize*=2;
    dom_file.write((char*)mLiquidWaterGraphs,totalgrapharraysize*sizeof(float));

    //Boundary data
    dom_file.write((char*)&mBoundCount,sizeof(int));
    dom_file.write((char*)mBoundTypes, mBoundCount*sizeof(int));
    dom_file.write((char*)mTemperatureLag, 12*mBoundCount*sizeof(float));
    dom_file.write((char*)mAlphaConvection, 12*mBoundCount*sizeof(float));
    dom_file.write((char*)mBoundValues, 12*mBoundCount*sizeof(float));


    //Interconnect data
    dom_file.write((char*)&mInterconnectCount,sizeof(int));
    for (int icIdx = 0; icIdx<mInterconnectCount; icIdx++){
        dom_file.write((char*)&(mInterconnects[icIdx]->mOwner), sizeof(int));
        dom_file.write((char*)&(mInterconnects[icIdx]->mSource), sizeof(int));
        dom_file.write((char*)&(mInterconnects[icIdx]->mSourceSide), sizeof(int));
        dom_file.write((char*)&(mInterconnects[icIdx]->mSourceH), sizeof(float));
        dom_file.write((char*)&(mInterconnects[icIdx]->mMshift), sizeof(int));
        dom_file.write((char*)&(mInterconnects[icIdx]->mNshift), sizeof(int));
    }

    //Heat Source data
    dom_file.write((char*)&mHeatSourceCount,sizeof(int));
    dom_file.write((char*)mHeatSourceDensity, 12*mHeatSourceCount*sizeof(float));

    dom_file.close();
}
*/
/*
void Domain3d::SaveStaticDataTxt(std::string txtFileName){


}*/


// ****** CLEANUP****

EXPORT_XX
void Domain3d::ReleaseResources(){
    //free( );
    for (int idx=0; idx<mInterconnectCount; idx++)
        mInterconnects[idx]->ReleaseResources(mLocationNode);
    delete [] mInterconnects;

    for (int idx=0;idx<mBlockCount; idx++)
        mBlocks[idx]->ReleaseResources();
    delete [] mBlocks;


    free(mLiqPartAtTr);
    free(mHMelt);
    free(mTemperatureGraph);
    free(mLiqidWaterGraphPtrs);

    free(mHeatSourceDensity);

    free(mBoundValues);
    free(mAlphaConvection);
    free(mTemperatureLag);
    free(mBoundTypes);

    free(mCavernBound);

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

}

EXPORT_XX
float Domain3d::GetFinishTime(void){
    return mFinishTime;
}

EXPORT_XX
int Domain3d::GetVolumeCount(){
    int count = 0;
    for (int bIdx=0; bIdx<mBlockCount; bIdx++)
        count+=mBlocks[bIdx]->GetVolumeCount();
    return count;
}

EXPORT_XX
int Domain3d::GetStepCount(){
  return mStepCounter;
}



// ************
/*
EXPORT_XX
int Domain3dMC::SetTimeToCompute(float timetocompute){
    mFinishTime = mTime + timetocompute;
    std::cout<<timetocompute<<std::endl;
    return 0;
}


EXPORT_XX
uint8_t Domain3dMC::GetMonth(void){
    return mMonth;
}

EXPORT_XX
float Domain3dMC::GetSavesPerMonth(void){
    return mSavesPerMonth;
}




*/
