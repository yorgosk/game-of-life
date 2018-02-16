/* File: gameOfLife_cuda.cu
  Authors: Kamaras Georgios
  Date:
*/
#include "header_cuda.h"

/* "play" the game of life for a given matrix -- compute the next generation */
/* source: https://devtalk.nvidia.com/default/topic/468304/linking-c-and-cuda-files-with-nvcc-and-gcc/ */
extern "C"
float playTheGame(char* initial, int rows, int columns) {
  /* device matrixes */
  char *dev_before = NULL, *dev_after = NULL, *temp = NULL, *dev_stepsTemp = NULL, *endFlagDev = NULL, *endFlagHost = NULL;
  int endFlag = 0, reps = 0;
  /* for time recording */
  float timer = 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  /* Allocate CPU memory for each host's matrix */
  endFlagHost = (char*) malloc(((rows*columns)/BLOCKSIZE)*sizeof(char));
  int i;
  for (i = 0; i < (rows*columns)/BLOCKSIZE; i++) endFlagHost[i] = -1;
  /* Allocate GPU memory for each device's matrix (2 simulation and 1 end-condition matrixes) */
  gpuErrorCheck(cudaMalloc((void**)&dev_before, rows*columns*sizeof(char)));
  gpuErrorCheck(cudaMalloc((void**)&dev_after, rows*columns*sizeof(char)));
  gpuErrorCheck(cudaMalloc((void**)&endFlagDev, ((rows*columns)/BLOCKSIZE)*sizeof(char)));

  /* Copy host matrixes to device memory matrixes */
  gpuErrorCheck(cudaMemcpy(dev_before, initial, rows*columns*sizeof(char), cudaMemcpyHostToDevice));
  int j;
  for (i = 0; i < rows; i++) for (j = 0; j < columns; j++) initial[columns*i + j] = 0;
  gpuErrorCheck(cudaMemcpy(dev_after, initial, rows*columns*sizeof(char), cudaMemcpyHostToDevice));
  gpuErrorCheck(cudaMemcpy(endFlagDev, endFlagHost, ((rows*columns)/BLOCKSIZE)*sizeof(char), cudaMemcpyHostToDevice));

  /* for similarity check every n-generations */
  gpuErrorCheck(cudaMalloc((void**)&dev_stepsTemp, rows*columns*sizeof(char)));
  gpuErrorCheck(cudaMemcpy(dev_stepsTemp, dev_before, rows*columns*sizeof(char), cudaMemcpyDeviceToDevice));

  /* grid dimension: number of blocks on each dim */
  dim3 gridDim(CEILING(columns, BLOCKSIZE), CEILING(rows, BLOCKSIZE));
  /* block dimension: block size on each dim */
  dim3 blockDim(BLOCKSIZE, BLOCKSIZE);

  /* for debugging */
  printf("\t\tgridDim1 = %d blockDim2 = %d\n", CEILING(rows, BLOCKSIZE), CEILING(columns, BLOCKSIZE));

  /* start recording */
  gpuErrorCheck(cudaEventRecord(start));

  /* Our simulation's loop */
  while (!endFlag/* && reps < MAXREPS*/) {
    endFlag = 1;
    /* call CUDA kernel's function with the desired dimensions */
    nextGenerationCUDAKernel<<<gridDim, blockDim>>>(dev_before, dev_after, rows, columns);
    /* check for end condition */
    for (i = 0; i < (rows*columns)/BLOCKSIZE; i++) endFlagHost[i] = -1;
    gpuErrorCheck(cudaMemcpy(endFlagDev, endFlagHost, ((rows*columns)/BLOCKSIZE)*sizeof(char), cudaMemcpyHostToDevice));
    endConditionCheck<<<gridDim, blockDim>>>(dev_before, dev_after, rows, columns, endFlagDev);
    gpuErrorCheck(cudaMemcpy(endFlagHost, endFlagDev, ((rows*columns)/BLOCKSIZE)*sizeof(char), cudaMemcpyDeviceToHost));
    for (i = 0; i < (rows*columns)/BLOCKSIZE; i++) {
      // printf("%d ", endFlagHost[i]);
      if (endFlagHost[i] == 0) {
        endFlag = 0;
        break;  // no need to loop further
      }
    }
    // printf("\n");
    /* swap simulation's matrixes */
    temp = dev_before;
    dev_before = dev_after;
    dev_after = temp;
#ifdef _NGENERATIONS_
    /* check for same generation every N steps to avoid looping between generations */
    if (!(reps % STEPS) && !endFlag) {
      endFlag = 1;
      for (i = 0; i < (rows*columns)/BLOCKSIZE; i++) endFlagHost[i] = -1;
      gpuErrorCheck(cudaMemcpy(endFlagDev, endFlagHost, ((rows*columns)/BLOCKSIZE)*sizeof(char), cudaMemcpyHostToDevice));
      nGenerationsCheck<<<gridDim, blockDim>>>(dev_before, dev_stepsTemp, rows, columns, endFlagDev);
      gpuErrorCheck(cudaMemcpy(endFlagHost, endFlagDev, ((rows*columns)/BLOCKSIZE)*sizeof(char), cudaMemcpyDeviceToHost));
      for (i = 0; i < (rows*columns)/BLOCKSIZE; i++) {
        // printf("%d ", endFlagHost[i]);
        if (endFlagHost[i] == 0) {
          endFlag = 0;
          break;  // no need to loop further
        }
      }
      // printf("----------\n");
      gpuErrorCheck(cudaMemcpy(dev_stepsTemp, dev_before, rows*columns*sizeof(char), cudaMemcpyDeviceToDevice));
    }
#endif
    /* for debugging */
    reps++;
  }

  /* stop recording */
  gpuErrorCheck(cudaEventRecord(stop));
#ifdef _DEBUGGING_
  /* for debugging */
  if (reps == MAXREPS && !endFlag) printf("\t\t\tMAXREPS REACHED\n");
  else printf("\t\t\tend-condition reached after %d repetitions\n", reps);
#endif
  /* get the last error from a runtime call */
  /* source: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html#group__CUDART__ERROR_1g3529f94cb530a83a76613616782bd233 */
  gpuErrorCheck(cudaGetLastError());
  /* blocks until the device has completed all preceding requested tasks */
  /* source: https://www.cs.cmu.edu/afs/cs/academic/class/15668-s11/www/cuda-doc/html/group__CUDART__THREAD_g6e0c5163e6f959b56b6ae2eaa8483576.html */
  gpuErrorCheck(cudaThreadSynchronize());

  /* Copy the right matrix back to host */
  gpuErrorCheck(cudaMemcpy(initial, dev_before, rows*columns*sizeof(char), cudaMemcpyDeviceToHost));

  /* Free dynamically allocated CPU memory */
  free(endFlagHost);
  /* Free dynamically allocated GPU memory */
  gpuErrorCheck(cudaFree(dev_before));
  gpuErrorCheck(cudaFree(dev_after));
  gpuErrorCheck(cudaFree(dev_stepsTemp));
  gpuErrorCheck(cudaFree(endFlagDev));

  /* calculate recorded time, after giving the GPU some time after the last cudaEvent...() call */
  gpuErrorCheck(cudaEventElapsedTime(&timer, start, stop));

  return timer;
}

/* compute the next generation in CUDA kernel, where each thread takes care of one element of the "before" matrix */
__global__ void nextGenerationCUDAKernel(char* before, char* after, int rows, int columns) {
  // printf("nextGenerationCUDAKernel\n");
  /* calculate coordinates in the topology */
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
#ifdef _DEBUGGING_
  /* for debugging */
  printf("\t\tx = %d y = %d element = %d\n", x, y, before[y*columns+x]);
#endif
  if ((x >= 0) && (x < columns - 1) && (y >= 0) && (y < rows - 1)) {
    /* compute the next generation */
    int north, east, south, west, north_west, north_east, south_west, south_east; // neighbors indexes
    int my_rank = columns*y + x + 1;  // a pseudo-rank to help us with the neighbors calculations
    
    if (y)
      north = my_rank - columns;
    else
      north = rows * columns - (columns - my_rank);
    if (my_rank % columns)
      east = my_rank + 1;
    else
      east = my_rank - columns + 1;
    if (my_rank + columns <= rows*columns)
      south = my_rank + columns;
    else
      south = my_rank % columns;
    if (x)
      west = my_rank - 1;
    else
      west = my_rank + columns - 1;
    if (((y) && (my_rank % columns)) || (!(y) && (my_rank % columns)))
      north_east = north + 1;
    else if ((y) && !(my_rank % columns))
      north_east = east - columns + 1;
    else
      north_east = columns * rows - columns + 1;
    if (((y) && (x)) || (!(y) && (x)))
      north_west = north - 1;
    else if ((y) && !(x))
      north_west =  west - columns;
    else
      north_west = columns * rows;
    if (((my_rank + columns <= rows*columns) && (x))
      || (!(my_rank + columns <= rows*columns) && (x)))
      south_west = south - 1;
    else if ((my_rank + columns <= rows*columns) && !(x))
      south_west = west + columns;
    else
      south_west = columns;
    if (((my_rank + columns <= rows*columns) && (my_rank % columns))
      || (!(my_rank + columns <= rows*columns) && (my_rank % columns)))
      south_east = south + 1;
    else if ((my_rank + columns <= rows*columns) && !(my_rank % columns))
      south_east = east + columns;
    else
      south_east = 1;
#ifdef _DEBUGGING_
    /* for debugging */
    printf("\n\tx = %d y = %d before[%d][%d] = %d\n\tnorth = %d east = %d south = %d west = %d north_west = %d north_east = %d south_east = %d south_west = %d\n",
            x, y, x, y, before[columns*x + y], north, east, south, west, north_west, north_east, south_east, south_west);
#endif
    /* calculate neighbors positions */
    int neighbors = before[north-1] + before[east-1] + before[south-1] + before[west-1]
                    + before[north_west-1] + before[north_east-1] + before[south_west-1] + before[south_east-1];
    if (before[columns*y+x] == ALIVE) {     // if there is an organism
      if (neighbors <= 1) after[columns*y+x] = DEAD;  // dies from loneliness
      else if (neighbors <= 3) after[columns*y+x] = ALIVE; // lives
      else after[columns*y+x] = DEAD;   // dies from overpopulation
    } else {               // if there is no organism
      if (neighbors == 3) after[columns*y+x] = ALIVE;  // a new organism is born
      else after[columns*y+x] = DEAD;   // no organism
    }
  }
}

/* check for the end condition
-- same matrix for 2 consecutive generations
-- or, there is no organism left alive */
__global__ void endConditionCheck(char* gen1, char* gen2, int rows, int columns, char* endFlag) {
  /* calculate coordinates in the topology */
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
#ifdef _DEBUGGING_
  /* for debugging */
  printf("\t\tx = %d y = %d element = %d blockID = %d threadIdx.x = %d threadIdx.y = %d element = %d\n",
          x, y, gen1[columns*x+y], blockId, threadIdx.x, threadIdx.y, gen1[columns*threadIdx.x + threadIdx.y]);
#endif
  /* use shared memory for the same generation and none-alive parts of end-condition check of the current thread's block */
  __shared__ char blockSame[BLOCKSIZE][BLOCKSIZE], blockDead[BLOCKSIZE][BLOCKSIZE];

  if ((x >= 0) && (x < columns - 1) && (y >= 0) && (y < rows - 1)) {
    /* check if same for two consecutive generations */
    if (gen1[columns*y + x] == gen2[columns*y + x])
      blockSame[threadIdx.x][threadIdx.y] = 1;
    else
      blockSame[threadIdx.x][threadIdx.y] = 0;
    /* check if currently dead */
    if (gen2[columns*y + x] == DEAD)
      blockDead[threadIdx.x][threadIdx.y] = 1;
    else
      blockDead[threadIdx.x][threadIdx.y] = 0;
    /* synchronize threads in block level */
    /* source: https://stackoverflow.com/questions/15240432/does-syncthreads-synchronize-all-threads-in-the-grid */
    __syncthreads();

    /* the first thread of each block, checks if block has reached end-condition */
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      int i, j;
      int localEndFlag = 1;
      // printf("\nblockSame:\n");
      for (i = 0; i < BLOCKSIZE; i++) {
        for (j = 0; j < BLOCKSIZE; j++) {
          // printf("%d ", blockSame[i][j]);
          if ((blockSame[i][j] != 1) && (blockSame[i][j] == 0)) {
            localEndFlag = 0;
            break; // no need to loop further
          }
        }
        if (!localEndFlag) break; // no need to loop further
        // printf("\n");
      }
      if (!localEndFlag) {
        localEndFlag = 1;
        // printf("\nblockDead:\n");
        for (i = 0; i < BLOCKSIZE; i++) {
          for (j = 0; j < BLOCKSIZE; j++) {
            // printf("%d ", blockDead[i][j]);
            if ((blockDead[i][j] != 1) && (blockDead[i][j] == 0)) {
              localEndFlag = 0;
              break; // no need to loop further
            }
          }
          if (!localEndFlag) break; // no need to loop further
          // printf("\n");
        }
      }

      endFlag[blockId] = localEndFlag;
    }
  }
}

/* check for repeated generations every N generations */
__global__ void nGenerationsCheck(char* gen1, char* gen2, int rows, int columns, char* endFlag) {
  /* calculate coordinates in the topology */
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
#ifdef _DEBUGGING_
  /* for debugging */
  printf("\t\tx = %d y = %d element = %d blockID = %d threadIdx.x = %d threadIdx.y = %d element = %d\n",
          x, y, gen1[columns*x+y], blockId, threadIdx.x, threadIdx.y, gen1[columns*threadIdx.x + threadIdx.y]);
#endif
  /* use shared memory for the same generation and none-alive parts of end-condition check of the current thread's block */
  __shared__ char blockSame[BLOCKSIZE][BLOCKSIZE];

  if ((x >= 0) && (x < columns - 1) && (y >= 0) && (y < rows - 1)) {
    /* check if same for two consecutive generations */
    if (gen1[columns*y + x] == gen2[columns*y + x])
      blockSame[threadIdx.x][threadIdx.y] = 1;
    else
      blockSame[threadIdx.x][threadIdx.y] = 0;
    /* synchronize threads in block level */
    /* source: https://stackoverflow.com/questions/15240432/does-syncthreads-synchronize-all-threads-in-the-grid */
    __syncthreads();

    /* the first thread of each block, checks if block has reached end-condition */
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      int i, j;
      int localEndFlag = 1;
      // printf("\nblockSame:\n\n");
      for (i = 0; i < BLOCKSIZE; i++) {
        for (j = 0; j < BLOCKSIZE; j++) {
          // printf("%d ", blockSame[i][j]);
          if ((blockSame[i][j] != 1) && (blockSame[i][j] == 0)) {
            localEndFlag = 0;
            break; // no need to loop further
          }
        }
        if (!localEndFlag) break; // no need to loop further
        // printf("\n");
      }

      endFlag[blockId] = localEndFlag;
    }
  }
}