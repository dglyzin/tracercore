/*
 * interconnect.h
 *
 *  Created on: Dec 25, 2014
 *      Author: dglyzin
 */

#ifndef INTERCONNECT_H_
#define INTERCONNECT_H_

class Interconnect{
    friend class Domain3d;
    friend class Block3dCpu;
    friend class Block3dCuda;

public:
    Interconnect(int owner, int source, int sourceSide, float sourceH, int mshift, int nshift,
            int ownerNode, int ownerDevice, int sourceNode, int sourceDevice, int mdim, int ndim){
        //necessary data
        mOwner = owner;
        mSource = source;
        mSourceSide = sourceSide;
        mSourceH = sourceH;
        mMshift = mshift;
        mNshift = nshift;
        //prepare convenience interconnect data
        mOwnerNode = ownerNode;
        mOwnerDevice = ownerDevice;
        mSourceNode = sourceNode;
        mSourceDevice = sourceDevice;
        mMdim = mdim;
        mNdim = ndim;
    }
    void PrepareData(int locationNode);
    void ReleaseResources(int locationNode);
    void SendRecv(int locationNode);

private:
    int mOwner;
    int mSource;
    int mSourceSide;
    float mSourceH;
    int mMshift;
    int mNshift;
    //convenience arrays
    int mMdim;
    int mNdim;
    int mOwnerNode;
    int mOwnerDevice;
    int mSourceNode;
    int mSourceDevice;
    //arrays to be updated every step
    float* mSourceConductivity;
    float* mSourceTemperature;
    float* mOwnerConductivity;
    float* mOwnerTemperature;
};

#endif /* INTERCONNECT_H_ */
