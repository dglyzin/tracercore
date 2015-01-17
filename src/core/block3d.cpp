/*
 * block3d.cpp
 *
 *  Created on: Nov 5, 2014
 *      Author: dglyzin
 */

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "block3d.h"
#include "approx.h"

// ******** LOADING ROUTINES *******************
void Block3d::LoadGeometryFromFile(std::ifstream &input_file){


    input_file.read ((char*)&mXoffset, sizeof(int));//useful for graphics output only
    input_file.read ((char*)&mYoffset, sizeof(int));
    input_file.read ((char*)&mZoffset, sizeof(int));

    input_file.read ((char*)&mXcount, sizeof(int));
    input_file.read ((char*)&mYcount, sizeof(int));
    input_file.read ((char*)&mZcount, sizeof(int));

    mZSliceSize = mYcount*mXcount;

    std::cout<<"Loading block"<<mXcount<<"x"<<mYcount<<"x"<<mZcount<<std::endl;

    mSizes[AXISX] = (float*) malloc(mXcount*sizeof(float));
    mSizes[AXISY] = (float*) malloc(mYcount*sizeof(float));
    mSizes[AXISZ] = (float*) malloc(mZcount*sizeof(float));
    input_file.read ((char*)mSizes[AXISX], mXcount*sizeof(float));
    input_file.read ((char*)mSizes[AXISY], mYcount*sizeof(float));
    input_file.read ((char*)mSizes[AXISZ], mZcount*sizeof(float));

}

void Block3d::LoadFromFile(std::ifstream &input_file_dom, std::ifstream &input_file_bin, int blockNum, int blockLocationNode, int blockLocationDevice){
    //input_file has domain info, input_file_bin contains state.
    //should be kept in block3dstub
    mSelfIdx = blockNum;
    mLocationNode = blockLocationNode;
    mLocationDevice = blockLocationDevice;

    LoadGeometryFromFile(input_file_dom);


    int volumescount = mXcount*mYcount*mZcount;

    //thefollowing should not be in block3dstub
    //in stub we should just skip:
    //input_file.seekg(volumescount*sizeof(float), input_file.cur);
    //input_file.seekg(volumescount*sizeof(float), input_file.cur);
    //input_file.seekg(volumescount*sizeof(char), input_file.cur);
    //input_file.seekg(volumescount*sizeof(float), input_file.cur);
    //input_file.seekg(volumescount*sizeof(uint16_t), input_file.cur);


    //state
    mTemperature = (float*) malloc(volumescount*sizeof(float));
    input_file_bin.read((char*)mTemperature,volumescount*sizeof(float));
    mEnthalpy = (float*) malloc(volumescount*sizeof(float));
    input_file_bin.read((char*)mEnthalpy,volumescount*sizeof(float));
    mAtPhaseTransition = (char*) malloc(volumescount*sizeof(char));
    input_file_bin.read((char*)mAtPhaseTransition,volumescount*sizeof(char));
    mLiquidPart = (float*) malloc(volumescount*sizeof(float));
    input_file_bin.read((char*)mLiquidPart,volumescount*sizeof(float));

    //properties
    uint16_t* tmp      = (uint16_t*) malloc(volumescount*sizeof(uint16_t));
    mBodyType = (uint8_t*) malloc(volumescount*sizeof(uint8_t));
    mBody    = (uint16_t*) malloc(volumescount*sizeof(uint16_t));
    input_file_dom.read((char*)tmp,volumescount*sizeof(uint16_t));
    for (int idx=0;idx<volumescount;idx++){
        mBody[idx] = tmp[idx]/2;
        mBodyType[idx]  = tmp[idx]%2;
    }

    free(tmp);

    mHeatSources = (uint16_t*) malloc(volumescount*sizeof(uint16_t));
    input_file_dom.read((char*)mHeatSources, volumescount*sizeof(uint16_t));

    mBorderSizeM[0] = mZcount;
    mBorderSizeN[0] = mYcount;
    mBorderSizeR[0] = mXcount;

    mBorderSizeM[1] = mZcount;
    mBorderSizeN[1] = mXcount;
    mBorderSizeR[1] = mYcount;

    mBorderSizeM[2] = mYcount;
    mBorderSizeN[2] = mXcount;
    mBorderSizeR[2] = mZcount;

    mBorderSideRidx[0] = 0;
    mBorderSideRidx[1] = mXcount;
    mBorderSideRidx[2] = 0;
    mBorderSideRidx[3] = mYcount;
    mBorderSideRidx[4] = 0;
    mBorderSideRidx[5] = mZcount;

    //x->r z->m y->n => (xCount+1)*yCount*zCount
    mBorderStrideM[0] = (mXcount+1)*mYcount;
    mBorderStrideN[0] = mXcount+1;
    mBorderStrideR[0] = 1;

    //y->r z->m x->n => xCount*(yCount+1)*zCount
    mBorderStrideM[1] = mXcount*(mYcount+1);
    mBorderStrideN[1] = 1;
    mBorderStrideR[1] = mXcount;

    //z->r y->m x->n => xCount*yCount*(zCount+1)
    mBorderStrideM[2] = mXcount;
    mBorderStrideN[2] = 1;
    mBorderStrideR[2] = mXcount*mYcount;


    //x->r z->m y->n => xCount*yCount*zCount
    mBodyStrideM[0] = mXcount*mYcount;
    mBodyStrideN[0] = mXcount;
    mBodyStrideR[0] = 1;

    //y->r z->m x->n => xCount*yCount*zCount
    mBodyStrideM[1] = mXcount*mYcount;
    mBodyStrideN[1] = 1;
    mBodyStrideR[1] = mXcount;

    //z->r y->m x->n => xCount*yCount*zCount
    mBodyStrideM[2] = mXcount;
    mBodyStrideN[2] = 1;
    mBodyStrideR[2] = mXcount*mYcount;


    for (int bIdx =0; bIdx<6; bIdx++){
        int axis = bIdx/2;
        int size = mBorderSizeM[axis]*mBorderSizeN[axis];
        tmp               = (uint16_t*) malloc(size*sizeof(uint16_t));
        mBorderType[bIdx] = (uint8_t*)  malloc(size*sizeof(uint8_t));
        mBorder[bIdx]     = (uint16_t*) malloc(size*sizeof(uint16_t));
        input_file_dom.read((char*)tmp, size*sizeof(uint16_t));
        for (int idx=0;idx<size; idx++){
            mBorderType[bIdx][idx] = tmp[idx] % 2;
            mBorder[bIdx][idx] = tmp[idx]/2;
        }
        free(tmp);
    }

}


//***** MISC ***********


float Block3d::GetMinH(){
    float minh;

    minh=mSizes[AXISX][0];
    for (int i = 1; i<(int)mXcount; i++)
        if (mSizes[AXISX][i]<minh)
            minh = mSizes[AXISX][i];
    for (int i = 0; i<(int)mYcount; i++)
        if (mSizes[AXISY][i]<minh)
            minh = mSizes[AXISY][i];
    for (int i = 0; i<(int)mZcount; i++)
        if (mSizes[AXISZ][i]<minh)
            minh = mSizes[AXISZ][i];
    return minh;
}

//major 2d axis
int Block3d::GetSideMsize(int side){
    if ((side == ZSTART) || (side == ZEND))
        return mYcount;
    else
        return mZcount;
}

//minor 2d axis
int Block3d::GetSideNsize(int side){
    if ((side == XSTART) || (side == XEND))
        return mYcount;
    else
        return mXcount;
}


// ********* COMPUTATIONS **************
//use known temperature to compute Enthalpy
//Needed only once
void Block3d::ComputeEnthalpy(float* liqPartAtTr, float** liqidWaterGraphPtrs, int* liquidWaterGraphLength, float* hMelt,
        float* thawedSoilVolHeat, float* frozenSoilVolHeat, float* drySoilDensity, float* waterPerDrySoilMass,
        float* phaseTransitionPoint){

#pragma omp parallel
    {
    int entindex, idxX, idxY, idxZ, soilidx;
    float ThCapacity,FrCapacity, HMelt, LiqPart, LiqPartAtTr;
    #pragma omp for
    for(idxZ=0; idxZ<(int)mZcount; idxZ++)
        for(idxY=0; idxY<(int)mYcount; idxY++)
            for(idxX=0; idxX<(int)mXcount; idxX++){
                entindex = idxZ*mZSliceSize+idxY*mXcount+idxX;
                soilidx = mBody[entindex];
                LiqPartAtTr = liqPartAtTr[soilidx];
                LiqPart = GetLiquidPartForSoilTemp(liqidWaterGraphPtrs[soilidx], liquidWaterGraphLength[soilidx], mTemperature[entindex]);
                HMelt = hMelt[soilidx];
                ThCapacity = thawedSoilVolHeat[soilidx];//mDrySoilDensity[soilindex]*(mDrySoilSpecHeat[soilindex]+mWaterSpecHeat*mWaterPerDrySoilMass[soilindex]);
                FrCapacity = frozenSoilVolHeat[soilidx];//mDrySoilDensity[soilindex]*(mDrySoilSpecHeat[soilindex]+(mIceVolumetricExpansion*mIceSpecHeat *(1-LiqPart)  + LiqPart*mWaterSpecHeat)*mWaterPerDrySoilMass[soilindex]);
                if (mAtPhaseTransition[entindex])
                    mEnthalpy[entindex] = mWaterLatentHeat*drySoilDensity[soilidx]*waterPerDrySoilMass[soilidx]*(mLiquidPart[entindex]-LiqPartAtTr);
                else if (mTemperature[entindex]>phaseTransitionPoint[soilidx])
                    mEnthalpy[entindex] = HMelt+ThCapacity*(mTemperature[entindex]-phaseTransitionPoint[soilidx]);
                else
                    mEnthalpy[entindex] = FrCapacity*(mTemperature[entindex]-phaseTransitionPoint[soilidx])+mWaterLatentHeat*drySoilDensity[soilidx]*waterPerDrySoilMass[soilidx]*(LiqPart-LiqPartAtTr);;
            }
    }

}

// ************* Saving routines ****************
void Block3d::SaveBlockDataBin(std::string fileNameBase){
    //std::ofstream dom_file((fileNameBase+std::string(".dom")).c_str(), std::ios::out | std::ofstream::app | std::ios::binary);
    std::ofstream bin_file((fileNameBase+std::string(".bin")).c_str(), std::ios::out | std::ofstream::app | std::ios::binary);
    /*
    dom_file.write((char*)&mLocationNode,sizeof(int));
    dom_file.write((char*)&mLocationDevice,sizeof(int));

    //offset
    dom_file.write((char*)&mXoffset, sizeof(int));
    dom_file.write((char*)&mYoffset, sizeof(int));
    dom_file.write((char*)&mZoffset, sizeof(int));

    //dimension
    dom_file.write((char*)&mXcount, sizeof(int));
    dom_file.write((char*)&mYcount, sizeof(int));
    dom_file.write((char*)&mZcount, sizeof(int));

    //grid sizes
    dom_file.write((char*)mXsizes, mXcount*sizeof(float));
    dom_file.write((char*)mYsizes, mYcount*sizeof(float));
    dom_file.write((char*)mZsizes, mZcount*sizeof(float));
*/
    int volumescount = mXcount*mYcount*mZcount;
    bin_file.write((char*)mTemperature,volumescount*sizeof(float));
    bin_file.write((char*)mEnthalpy,volumescount*sizeof(float));
    bin_file.write((char*)mAtPhaseTransition,volumescount*sizeof(char));
    bin_file.write((char*)mLiquidPart,volumescount*sizeof(float));
/*
    dom_file.write((char*)mSoilTypes,volumescount*sizeof(uint16_t));
    dom_file.write((char*)mHeatSources, volumescount*sizeof(uint16_t));

    dom_file.write((char*)mBorderX0, mZcount*mYcount*sizeof(uint16_t));
    dom_file.write((char*)mBorderXMax, mZcount*mYcount*sizeof(uint16_t));
    dom_file.write((char*)mBorderY0,mXcount*mZcount*sizeof(uint16_t));
    dom_file.write((char*)mBorderYMax,mXcount*mZcount*sizeof(uint16_t));
    dom_file.write((char*)mBorderZ0,mXcount*mYcount*sizeof(uint16_t));
    dom_file.write((char*)mBorderZMax,mXcount*mYcount*sizeof(uint16_t));
*/
    //dom_file.close();
    bin_file.close();
}


void Block3d::SaveBlockDataTxt(std::string txtFileName){
    std::ofstream res_file(txtFileName.c_str(), std::ios::out | std::ofstream::app);
    res_file<<"Block #"<< mSelfIdx <<std::endl;
    res_file<<"Node "<< mLocationNode <<", device "<< mLocationDevice<<std::endl;
    res_file<<"Offset: "<< mXoffset << ", "<<mYoffset<<", "<<mZoffset <<std::endl;
    res_file<<"Grid dimension: "<< mXcount << ", "<<mYcount<<", "<<mZcount <<std::endl;
    res_file<<"Grid size X: ";
    for(int idx=0; idx<mXcount; idx++)
       res_file << mSizes[AXISX][idx]<<" ";
    res_file <<std::endl;
    res_file<<"Grid size Y: ";
    for(int idx=0; idx<mYcount; idx++)
       res_file << mSizes[AXISY][idx]<<" ";
    res_file <<std::endl;
    res_file<<"Grid size Z: ";
    for(int idx=0; idx<mZcount; idx++)
       res_file << mSizes[AXISZ][idx]<<" ";
    res_file <<std::endl;

    res_file<<"Temperature"<<std::endl;
    for(int idxZ=0; idxZ<mZcount; idxZ++){
        for(int idxY=0; idxY<mYcount; idxY++)
        //int idxY = 0;
        {
            res_file<<idxZ<<" "<<idxY<<": ";
            for(int idxX=0; idxX<mXcount; idxX++)
                res_file << mTemperature[idxZ*mZSliceSize+idxY*mXcount+idxX]<<" ";
            res_file<<std::endl;
        }
        res_file<<std::endl;
    }

   /* res_file<<"Enthalpy"<<std::endl;
    for(int idxZ=0; idxZ<mZcount; idxZ++){
        for(int idxY=0; idxY<mYcount; idxY++)
        {
            res_file<<idxZ<<" "<<idxY<<": ";
            for(int idxX=0; idxX<mXcount; idxX++)
                res_file << mEnthalpy[idxZ*mZSliceSize+idxY*mXcount+idxX]<<" ";
            res_file<<std::endl;
        }
        res_file<<std::endl;
    }

    res_file<<"At phase transition"<<std::endl;
    for(int idxZ=0; idxZ<mZcount; idxZ++){
        for(int idxY=0; idxY<mYcount; idxY++)
        {
            res_file<<idxZ<<" "<<idxY<<": ";
            for(int idxX=0; idxX<mXcount; idxX++)
                res_file << (int)(mAtPhaseTransition[idxZ*mZSliceSize+idxY*mXcount+idxX])<<" ";
            res_file<<std::endl;
        }
        res_file<<std::endl;
    }

    res_file<<"LiquidPart"<<std::endl;
    for(int idxZ=0; idxZ<mZcount; idxZ++){
        for(int idxY=0; idxY<mYcount; idxY++)
        {
            res_file<<idxZ<<" "<<idxY<<": ";
            for(int idxX=0; idxX<mXcount; idxX++)
                res_file << mLiquidPart[idxZ*mZSliceSize+idxY*mXcount+idxX]<<" ";
            res_file<<std::endl;
        }
        res_file<<std::endl;
    }

    res_file<<"Soil type"<<std::endl;
    for(int idxZ=0; idxZ<mZcount; idxZ++){
        for(int idxY=0; idxY<mYcount; idxY++)
        {
            res_file<<idxZ<<" "<<idxY<<": ";
            for(int idxX=0; idxX<mXcount; idxX++)
                res_file << mSoilTypes[idxZ*mZSliceSize+idxY*mXcount+idxX]<<" ";
            res_file<<std::endl;
        }
        res_file<<std::endl;
    }

    res_file<<"X0 bound"<<std::endl;
        for(int idxZ=0; idxZ<mZcount; idxZ++){
            res_file<<idxZ<<": ";
            for(int idxY=0; idxY<mYcount; idxY++){
                res_file << mBoundX0[idxZ*mYcount+idxY]<<" ";

            }
            res_file<<std::endl;
        }
        res_file<<"Xmax bound"<<std::endl;
                for(int idxZ=0; idxZ<mZcount; idxZ++){
                    res_file<<idxZ<<": ";
                    for(int idxY=0; idxY<mYcount; idxY++){
                        res_file << mBoundXMax[idxZ*mYcount+idxY]<<" ";

                    }
                    res_file<<std::endl;
                }

        res_file<<"Y0 bound"<<std::endl;
                for(int idxZ=0; idxZ<mZcount; idxZ++){
                    res_file<<idxZ<<": ";
                    for(int idxX=0; idxX<mXcount; idxX++){
                        res_file << mBoundY0[idxZ*mXcount+idxX]<<" ";

                    }
                    res_file<<std::endl;
                }

        res_file<<"Ymax bound"<<std::endl;
                for(int idxZ=0; idxZ<mZcount; idxZ++){
                    res_file<<idxZ<<": ";
                    for(int idxX=0; idxX<mXcount; idxX++){
                        res_file << mBoundYMax[idxZ*mXcount+idxX]<<" ";

                    }
                    res_file<<std::endl;
                }

        res_file<<"Z0 bound"<<std::endl;
                for(int idxY=0; idxY<mYcount; idxY++){
                    res_file<<idxY<<": ";
                    for(int idxX=0; idxX<mXcount; idxX++){
                        res_file << mBoundZ0[idxY*mXcount+idxX]<<" ";

                    }
                    res_file<<std::endl;
                }


        res_file<<"Zmax bound"<<std::endl;
                for(int idxY=0; idxY<mYcount; idxY++){
                    res_file<<idxY<<": ";
                    for(int idxX=0; idxX<mXcount; idxX++){
                        res_file << mBoundZMax[idxY*mXcount+idxX]<<" ";

                    }
                    res_file<<std::endl;
                }
*/



    res_file.close();
/*
    res_file<<"Reciprocal Conductivity"<<std::endl;
    for(int idxZ=0; idxZ<(int)mZcount; idxZ++){
        for(int idxY=0; idxY<(int)mYcount; idxY++)
        //int idxY=0;
        {
            res_file<<idxZ<<" "<<idxY<<": ";
            for(int idxX=0; idxX<(int)mXcount; idxX++)
                res_file << mRecConductivity[idxZ*mZSliceSize+idxY*mXcount+idxX]<<" ";
            res_file<<std::endl;
        }
    }

    res_file<<"HEAT FLOW X"<<std::endl;
    for(int idxZ=0; idxZ<(int)mZcount; idxZ++){
        for(int idxY=0; idxY<(int)mYcount; idxY++)
        //int idxY=0;
        {
            for(int idxX=0; idxX<(int)mXcount+1; idxX++){
                int hflowindex = idxX + idxY*(mXcount+1) + idxZ*(mXcount+1)*mYcount;
                res_file << mHeatFlowX[hflowindex]<<" ";
            }
            res_file<<std::endl;
        }
    }


    res_file<<"HEAT FLOW Y"<<std::endl;
    for(int idxZ=0; idxZ<(int)mZcount; idxZ++){
        for(int idxY=0; idxY<(int)mYcount+1; idxY++)
        //int idxY=0;
        {
            for(int idxX=0; idxX<(int)mXcount; idxX++){
                int hflowindex = idxX+idxY*mXcount+idxZ*(mYcount+1)*mXcount;
                res_file << mHeatFlowY[hflowindex]<<" ";
            }
            res_file<<std::endl;
        }
    }


    res_file<<"HEAT FLOW Z"<<std::endl;
    for(int idxZ=0; idxZ<(int)mZcount+1; idxZ++){
        for(int idxY=0; idxY<(int)mYcount; idxY++)
        //int idxY=0;
        {
            for(int idxX=0; idxX<(int)mXcount; idxX++){
                int hflowindex = idxX+idxY*mXcount+idxZ*mYcount*mXcount;
                res_file << mHeatFlowZ[hflowindex]<<" ";
            }
            res_file<<std::endl;
        }
    }
*/
}

void Block3d::ReleaseBasicResources(){
//Block3d (anchestor) - allocated
    free(mCavernBound);
    for (int bIdx =0; bIdx<6; bIdx++){
        free(mBorderType[bIdx]);
        free(mBorder[bIdx]);
    }
    free(mHeatSources);
    free(mBodyType);
    free(mBody);
    free(mLiquidPart);
    free(mAtPhaseTransition);
    free(mEnthalpy);
    free(mTemperature);
    free(mSizes[AXISX] );
    free(mSizes[AXISY] );
    free(mSizes[AXISZ] );
}
