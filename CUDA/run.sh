#!/bin/bash

# Simple usage case: compilation, test-run and clean-up

make
./gameOfLife_cuda -f ../tests/test2x2.txt -r 2 -c 2
make clean
