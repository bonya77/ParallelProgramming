#!/bin/bash

# Компиляция всех программ с OpenMP
g++ -fopenmp -O2 -o p_variant1Dynamic p_variant1Dynamic.cpp
g++ -fopenmp -O2 -o p_variant1Guided p_variant1Guided.cpp
g++ -fopenmp -O2 -o p_variant1Static p_variant1Static.cpp
g++ -fopenmp -O2 -o p_variant2Dynamic p_variant2Dynamic.cpp
g++ -fopenmp -O2 -o p_variant2Guided p_variant2Guided.cpp
g++ -fopenmp -O2 -o p_variant2Static p_variant2Static.cpp

echo "Компиляция завершена"