/* File: header_cuda.h
   Authors: Kamaras Georgios
   Date:
*/
#ifndef __HEADER_CUDA__
#define __HEADER_CUDA__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

/* debugging */
#include <assert.h>
#define MAXREPS 100000

/* problem representation */
#define DEAD 0
#define ALIVE 1
#define BLOCKSIZE 2

/* default dimensions */
#define DEFAULT_ROWS 32
#define DEFAULT_COLUMNS 32

/* more optional functionality */
/* check for same generation every N steps to avoid looping between generations */
#define STEPS 5
#define _NGENERATIONS_
// #define _DEBUGGING_

/* ceiling function, for task division */
#define CEILING(a, b) ((a + b - 1) / b)

/* check for errors using the CUDA runtime API */
/* source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api */
#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      exit(code);
   }
}

/* For functions_cuda.c */

/* Various functions that we use throughout our CUDA implementation */
/* Initialize problem's representation */
void initializeRepresentation(int, char**, char**, int*, int*);
/* read matrix's initial state from a text-file */
void readInitialState(char*, char*, int, int);
/* set a random initial state for the matrix */
void setRandomInitialState(char*, int, int);
/* print problem's matrix -- for debugging */
void printMatrix(char*, int, int);

/* For gameOfLife_cuda.cu */

/* "play" the game of life for a given matrix -- compute the next generation */
/* source: https://devtalk.nvidia.com/default/topic/468304/linking-c-and-cuda-files-with-nvcc-and-gcc/ */
#ifdef __cplusplus
extern "C"
#endif
float playTheGame(char*, int, int);
/* compute the next generation in CUDA kernel, where each thread takes care of one element of the matrix */
__global__ void nextGenerationCUDAKernel(char*, char*, int, int);
/* check for the end condition
-- same matrix for 2 consecutive generations
-- or, there is no organism left alive */
__global__ void endConditionCheck(char*, char*, int, int, char*);
/* check for repeated generations every N generations */
__global__ void nGenerationsCheck(char*, char*, int, int, char*);

#endif
