/*
 * interconnect.cpp
 *
 *  Created on: Dec 25, 2014
 *      Author: dglyzin
 */
#include "interconnect.h"
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime_api.h>

void Interconnect::PrepareData(int locationNode){
    /*
     * Depending on whether this interconnect's sender or receiver is local
     * we allocate buffers
     */
    if (locationNode == mSourceNode && locationNode == mOwnerNode){
        //Data transfer within one node
        //cases: cpu->same cpu
        //       cpu->GPU
        //       GPU->cpu
        //       GPU->same GPU
        //       GPU->different GPU
        if (mSourceDevice==0 && mOwnerDevice==0){
            //cpu->same cpu
            mSourceConductivity = (float*) malloc(mMdim*mNdim*sizeof(float));
            mSourceTemperature = (float*) malloc(mMdim*mNdim*sizeof(float));
            mOwnerConductivity = mSourceConductivity;
            mOwnerTemperature = mSourceTemperature;
        }
        else if (mSourceDevice==mOwnerDevice && mOwnerDevice>0) {
            //GPU->same GPU
            cudaSetDevice(mSourceDevice-1);
            cudaMalloc((void**)&mSourceConductivity, mMdim*mNdim*sizeof(float));
            cudaMalloc((void**)&mSourceTemperature, mMdim*mNdim*sizeof(float));
            mOwnerConductivity = mSourceConductivity;
            mOwnerTemperature = mSourceTemperature;
        }
        else if ( (mSourceDevice==0 && mOwnerDevice>0) || (mSourceDevice>0 && mOwnerDevice==0) ) {
            //cpu->GPU->cpu
            cudaMallocHost((void**)&mSourceConductivity, mMdim*mNdim*sizeof(float));
            cudaMallocHost((void**)&mSourceTemperature, mMdim*mNdim*sizeof(float));
            mOwnerConductivity = mSourceConductivity;
            mOwnerTemperature = mSourceTemperature;
        }
        else if (mSourceDevice>0 && mOwnerDevice>0 && mOwnerDevice!=mSourceDevice)  {
            //GPU->different GPU
            cudaSetDevice(mSourceDevice-1);
            cudaMalloc((void**)&mSourceConductivity, mMdim*mNdim*sizeof(float));
            cudaMalloc((void**)&mSourceTemperature, mMdim*mNdim*sizeof(float));
            cudaSetDevice(mOwnerDevice-1);
            cudaMalloc((void**)&mOwnerConductivity, mMdim*mNdim*sizeof(float));
            cudaMalloc((void**)&mOwnerTemperature, mMdim*mNdim*sizeof(float));
        }
        else
            assert(false);
    }
    else{
        //data transfers between nodes:
        // cpu->cpu
        // cpu->GPU
        // GPU->cpu
        // GPU->GPU
        assert(false);
    }

}

void Interconnect::ReleaseResources(int locationNode){
    if (locationNode == mSourceNode && locationNode == mOwnerNode){
        //Data transfer within one node
        //cases: cpu->same cpu
        //       cpu->GPU
        //       GPU->cpu
        //       GPU->same GPU
        //       GPU->different GPU
        if (mSourceDevice==0 && mOwnerDevice==0){
            //cpu->same cpu
            free(mSourceConductivity);
            free(mSourceTemperature);

        }
        else if (mSourceDevice==mOwnerDevice && mOwnerDevice>0) {
            //GPU->same GPU
            cudaFree(mSourceConductivity);
            cudaFree(mSourceTemperature);
        }
        else if ( (mSourceDevice==0 && mOwnerDevice>0) || (mSourceDevice>0 && mOwnerDevice==0) ) {
            //cpu->GPU->cpu
            cudaFreeHost(mSourceConductivity);
            cudaFreeHost(mSourceTemperature);
        }
        else if (mSourceDevice>0 && mOwnerDevice>0 && mOwnerDevice!=mSourceDevice)  {
            //GPU->different GPU
            cudaFree(mSourceConductivity);
            cudaFree(mSourceTemperature);
            cudaFree(mOwnerConductivity);
            cudaFree(mOwnerTemperature);
        }
        else
            assert(false);
    }
    else{
        //data transfers between nodes:
        // cpu->cpu
        // cpu->GPU
        // GPU->cpu
        // GPU->GPU
        assert(false);

    }
}



void Interconnect::SendRecv(int locationNode){
    /*
     * locationNode is a node, where this method is called.
     * For every inter-node interconnect this method will be called twice - by sending and receiving nodes with corresponding locationNode
     * for intranode interconnects this method will be called once and should perform everything
     *
     */
    if (locationNode == mSourceNode && locationNode == mOwnerNode){
        //Data transfer within one node
        //cases: cpu->same cpu
        //       cpu->GPU
        //       GPU->cpu
        //       GPU->same GPU
        //       GPU->different GPU
        if (mSourceDevice==0 && mOwnerDevice==0){
            //cpu->same cpu
            //nothing to be done

        }
        else if (mSourceDevice==mOwnerDevice && mOwnerDevice>0) {
            //GPU->same GPU
            //nothing to be done
        }
        else if ( (mSourceDevice==0 && mOwnerDevice>0) || (mSourceDevice>0 && mOwnerDevice==0) ) {
            //cpu->GPU->cpu
            //nothing to be done as gpudirect is pinned memory is accessible
        }
        else if (mSourceDevice>0 && mOwnerDevice>0 && mOwnerDevice!=mSourceDevice)  {
            //GPU->different GPU
            cudaMemcpy(mOwnerConductivity, mSourceConductivity, mMdim*mNdim*sizeof(float), cudaMemcpyDefault);
            cudaMemcpy(mOwnerTemperature, mSourceTemperature, mMdim*mNdim*sizeof(float), cudaMemcpyDefault);
        }
        else
            assert(false);
    }
    else{
        //data transfers between nodes:
        // cpu->cpu
        // cpu->GPU
        // GPU->cpu
        // GPU->GPU
        assert(false);

    }
}
