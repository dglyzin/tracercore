/*
 * approx.cpp
 *
 *  Created on: Nov 9, 2014
 *      Author: dglyzin
 */

#include "mystdint.h"

float GetLiquidPartForSoilTemp(float* pDependence, int deplength, float temperature){
    float liqpart;
    //float* pDependence = mLiqidWaterGraphPtrs[soilindex];
    //int deplength = mLiquidWaterGraphLength[soilindex];
    //pDependence format: t0 v0 t1 v1 t2 v2....t(deplength-1) v(deplength-1)
    if (deplength<2)
        return -100.f;

    int leftpoint = 1;
    //std::cout<<deplength<<" "<< leftpoint <<std::endl;
    while ((temperature<pDependence[2*leftpoint])&&(leftpoint<deplength-1))
        leftpoint++;
    int rightpoint = leftpoint-1;
    //std::cout<<deplength<<" "<< leftpoint<<" " <<rightpoint  <<std::endl;
    liqpart =  (temperature-pDependence[2*leftpoint])/(pDependence[2*rightpoint]-pDependence[2*leftpoint])*(pDependence[2*rightpoint+1]-pDependence[2*leftpoint+1])+pDependence[2*leftpoint+1];
    if (liqpart>1)
        return 1.0f;
    else if (liqpart<0)
        return 0.0f;
    else
        return liqpart;

}
//Computes temperature of a given soil with given volumetric enthalpy
float GetTempForSoilEnt(float* tDependence, float enthalpy){
    //
    //float* tDependence = mTemperatureGraph + TEMP_GRAPH_LEN*2*soilindex;
    int deplength = TEMP_GRAPH_LEN;
    //tDependence format: e0 t0 e1 t1 e2 t2....e(deplength-1) t(deplength-1)
    int leftpoint = 1;
    while ((enthalpy<tDependence[2*leftpoint])&&(leftpoint<deplength-1))
        leftpoint++;
    int rightpoint = leftpoint-1;
    return (enthalpy-tDependence[2*leftpoint])/(tDependence[2*rightpoint]-tDependence[2*leftpoint])*(tDependence[2*rightpoint+1]-tDependence[2*leftpoint+1])+tDependence[2*leftpoint+1];

}
