/* File: header_mpi_openmp.h
   Author: Kamaras Georgios
   Date: 8/10/2017
*/
#ifndef __HEADER_MPI_OPENMP__
#define __HEADER_MPI_OPENMP__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>

#include <mpi.h>
#include <omp.h>

/* debugging */
#include <assert.h>
#define MAXREPS 5000

/* problem representation */
#define DEAD 0
#define ALIVE 1

/* default dimensions */
#define DEFAULT_ROWS 32
#define DEFAULT_COLUMNS 32

/* more optional functionality */
/* check for same generation every N steps to avoid looping between generations */
// #define STEPS 5
// #define _NGENERATIONS_
// #define _DEBUGGING_

/* Various functions that we use throughout our MPI implementation */
/* Initialize problem's representation */
void initializeRepresentation(int, char**, char**, int*, int*);
/* divide problem's matrix so that it's parts can be processed in parallel as efficiently as possible */
int divideWorkload(int, int, int);
/* read matrix's initial state from a text-file */
void readInitialState(char*, char*, int, int, int, int, int, int);
/* set a random initial state for the matrix */
void setRandomInitialState(char*, int, int);
/* "play" the game of life for a specific part of a matrix -- compute the next generation */
int nextGeneration(char*, char*, int, int, int, int, int);
/* check for the same matrix for 2 consecutive generations */
char sameGenerations(char*, char*, int, int);
/* check if there is no organism left alive */
char noneAlive(char*, int, int);
/* Get pointer to internal matrix position -- utility */
char *locate(char*, int, int, int);
/* print the sub-matrix and the process that it belongs to -- debugging */
void printSubmatrix(int, char*, int, int);

#endif
