#!/bin/bash

i=1
for a in "$@"; do
    echo "$i: $a"
    ((i++))
done


#if [ $1 != "salloc" -a $1 != "srun" ]
#then
#  echo Ошибка!
#  echo Первый аргумент может иметь только два значения "(salloc / srun)".
#  exit
#fi

#if [ $1 == "salloc" ]
#then
#  salloc -N$2 -n$2 mpirun bin/HS $3 $4 $5 $6 $7
#fi
#
#if [ $1 == "srun" ]
#then
#  srun -N$2 -n$2 bin/HS $3 $4 $5 $6 $7
#fi

