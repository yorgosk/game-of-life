#!/bin/bash

# Simple usage case: compilation, test-run and clean-up

make
mpirun -np 2 ./gameOfLife_mpi_openmp -f ../tests/test2x2.txt -r 2 -c 2
make clean
