#!/bin/bash

# Simple usage case: compilation, test-run and clean-up

#mpicc -g -Wall -o gameOfLife_mpi gameOfLife_mpi.c
make
mpiexec -np 2 ./gameOfLife_mpi -f ../tests/test2x2.txt -r 2 -c 2
make clean
