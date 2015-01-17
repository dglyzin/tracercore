/*
 * approx.h
 *
 *  Created on: Nov 9, 2014
 *      Author: dglyzin
 */

#ifndef APPROX_H_
#define APPROX_H_


float GetLiquidPartForSoilTemp(float* pDependence, int deplength, float temperature);
float GetTempForSoilEnt(float* tDependence, float enthalpy);


#endif /* APPROX_H_ */
