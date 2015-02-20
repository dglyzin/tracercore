#!/bin/bash

if [ $# -ne 4 ]
then
  echo Wrong number of arguments!
  exit
fi

if [ $1 == "salloc" ]
then
  salloc -N$2 -n$2 mpirun bin/HS $3 $4
fi

if [ $1 == "srun" ]
then
  srun -N$2 -n$2 bin/HS $3 $4
fi