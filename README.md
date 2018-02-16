# Parallel Systems -- Game of Life Implementation

## About

  This is a **Parallel Systems** project, developed at **Summer of 2017** by
  **Kamaras Georgios** for the Parallel Systems _(Programming)_ course.
  The goal of this project was designing, implementing and evaluating parallel programs in *MPI*, *MPI+OpenMp* and *Cuda* environments, by implementing the famous **[Conway's Game of Life]**.

  [Conway's Game of Life]: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life?oldformat=true

## Structure (Parts and Content)

* Part 1: Introduction
* Part 2: Data Sharing Design (sharing in _Blocks_, not lines, communication, processes' topology, etc)
* Part 3: *MPI* code design and implementation with the goal of reducing inert time or unnecessary calculations (communication choices, communication overlapping with calculations, avoiding many copies by using data-types, etc). Check for lack of matrix change after _n_ iterations.
* Part 4: Measuring running time, calculating speedup, efficiency and presentation of results. Observation of data and processors scaling. Constant number of repeats. Behavior demonstration using *Paraver*.
* Part 5: Adding *OpenMp* commands for parallelization of calculations (e.g. inner elements), so that a _hybrid_ program is developed. Observation of data and processors scaling.
* Part 6: Autonomous *Cuda* program with the same calculations.
* Part 7: Conclusions

## Usage

* Input initial state from file

  > ./gameOfLife_mpi -f [inputTextFile].txt -r [numOfRows] -c [numOfColumns]

* Random initial state

  > ./gameOfLife_mpi

* Similarly

  > ./gameOfLife_mpi_openmp -f [inputTextFile].txt -r [numOfRows] -c [numOfColumns]

  _or_

  > ./gameOfLife_mpi_openmp

  And

  > ./gameOfLife_cuda -f [inputTextFile].txt -r [numOfRows] -c [numOfColumns]

  _or_

  > ./gameOfLife_cuda

* For more, please take a look at the _run.sh_ scripts that come with each
  implementation.

### Input text-file sample

* We want to represent a matrix that has four organisms in his first line and two
  organisms (neighbors to at least one of the above) in the second line

  > 1 2  
  > 2 2  
  > 1 3  
  > 2 3  
  > 1 4  
  > 1 5  

## Contact & Feedback Details

* Kamaras Georgios: <sdi1400058@di.uoa.gr>
