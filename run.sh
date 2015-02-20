#!/bin/bash

if [ $# -ne 4 ]
then
  echo Неверное количество аргументов.
  echo Необходимо использовать 4 аргумента для запуска.
  echo 1. Тип запуска: "salloc" или "srun".
  echo 2. Количество используемых узлов.
  echo 3. Имя входного файла.
  echo 4. Имя выходного файла.
  exit
fi

if [ $1 != "salloc" -a $1 != "srun" ]
then
  echo Ошибка!
  echo Первый аргумент может иметь только два значения "(salloc / srun)".
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