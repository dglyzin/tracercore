#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "core/domain3d.h"
#include <omp.h>


int runTest(std::string namebase, int ttc_default, float ttc){
	double time1, time2;
    std::string outputfile(namebase);
	Domain3d Laputa;
	Laputa.LoadFromBinaryFile(namebase, 0);
	printf("Start!\n");
	time1 = omp_get_wtime();

	float finishTime = Laputa.GetFinishTime();
	if (!ttc_default)
	    finishTime = Laputa.GetTime()+ttc;

	while (Laputa.GetTime()<finishTime){
	    //std::cout << "System time = " << Laputa.GetTime() << " of " <<finishTime << std::endl;
	    Laputa.ProcessOneStep();
	}
    time2 = omp_get_wtime();

	std::cout << "Time: "<< time2-time1 <<std::endl;
    double bps = (double) Laputa.GetVolumeCount()*Laputa.GetStepCount()/(time2-time1)/1000000;
    std::cout << "Domain contains "<<Laputa.GetVolumeCount()<<" volumes. Steps completed: "<<Laputa.GetStepCount()<< std::endl;
    std::cout << "Performance: " << bps << " Mblocks per second" << std::endl<< std::endl;
    Laputa.SaveToFile(outputfile, 1);
    Laputa.ReleaseResources();
    printf("Finished!\n");

	/*if (!ttc_default)
	  LaputaMC.SetTimeToCompute(ttc);
	
	//totalSave(&Laputa, namebase+"MP");
	float finaltime = LaputaMC.GetFinishTime();
	
	int month = LaputaMC.GetMonth();
	float lastSave = LaputaMC.GetTime();
	float saveinterval = (float) (int) 30.0/LaputaMC.GetSavesPerMonth();
	int savesLeft = (int)LaputaMC.GetSavesPerMonth()-1;
    //std::cout<<"Number of saves per month asked" <<  Laputa.GetSavesPerMonth() << std::endl;


	time1 = omp_get_wtime();
	while (LaputaMC.GetTime()<finaltime){	
	  LaputaMC.ProcessOneStepCPU();

	  if (month!=LaputaMC.GetMonth()){
		  month = LaputaMC.GetMonth();
		  //std::cout<<"month changed " << month <<" " << Laputa.GetTime()<<std::endl;
		  lastSave = LaputaMC.GetTime();
		  savesLeft = (int)LaputaMC.GetSavesPerMonth()-1;
		  //save here
          LaputaMC.totalSave(namebase+"MC",savetext);
	  }

	  if (LaputaMC.GetTime()-lastSave>=saveinterval){
		  if (savesLeft>0){
			  savesLeft--;
		      lastSave = LaputaMC.GetTime();
			  //save here
              LaputaMC.totalSave(namebase+"MC",savetext);
		  }
	  }	  	  
	}
	time2 = omp_get_wtime();
	
    LaputaMC.finalSave(namebase,savetext);
*/
	return 0;
}




int main(int argc, char * argv[]){

	printf("Welcome to the Heat Distribution 3D Core\n");
	if (argc<2){
		std::cout<<"Please enter input file as command line argument." << std::endl;
		return 1;
	}	
	
	std::string namebase(argv[1]);

	
	float ttc =0.f;
	int ttc_default = 1;
    if (argc>2){
        ttc = (float)atof(argv[2]);
        ttc_default = 0;
    }

    runTest(namebase, ttc_default, ttc);

	return 0;
}
