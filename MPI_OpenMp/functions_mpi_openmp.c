/* File: functions_mpi.c
   Author: Kamaras Georgios
   Date: 8/10/2017
*/
#include "header_mpi_openmp.h"

/* Initialize problem's representation */
void initializeRepresentation(int argc, char** argv, char** filename, int* rows, int* columns) {
  /* Set-up based on command line's arguments */
  if (argc == 1) {  /* various things can be done if no arguments, I have left uncommented what works the best */
   //  srand(time(NULL));
   //  int possibleDimensions[6] = {2, 3, 4, 5, 6, 7};  // for 4, 9, 16, 25, 36 and 49 squares respectively
    // int randomIndex = rand() % 6;
   //  *rows = possibleDimensions[randomIndex];
    // *columns = *rows;
    *rows = DEFAULT_ROWS;
    *columns = DEFAULT_COLUMNS;
    // fprintf(stderr, "Usage: mpiexec -n <np> ./gameOfLife_mpi -f <filepath> -r <no.rows> -c <no.columns> .\nPlease, take a look at run.sh for simple usage case.\n");
    // exit(EXIT_FAILURE);
  } else if (argc == 7) {
    int i, rflag = 0, cflag = 0, fflag = 0;
    for (i = 1; i < argc-1; i += 2) {   // go through the program's arguments list
      // printf("%s\n", argv[i]);
      if (!fflag && !strcmp(argv[i], "-f")) { // the first time we find the file flag
        fflag = 1;
        *filename = malloc(strlen(argv[i+1])+1);
        strcpy(*filename, argv[i+1]);
      } else if (!rflag && !strcmp(argv[i], "-r")) { // the first time we find the rows flag
        rflag = 1;
        *rows = atoi(argv[i+1]);
      } else if (!cflag && !strcmp(argv[i], "-c")) { // the first time we find the columns flag
        cflag = 1;
        *columns = atoi(argv[i+1]);
      } else {
        fprintf(stderr, "ERROR! Bad argument formation!\n");
        /* abort MPI and terminate reporting failure */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE);
      }
    }
    // printf("%s %d %d\n", *filename, *rows, *columns);
  } else {
    /* miss-program */
    fprintf(stderr, "%s: Error: Insufficient number of arguments\n", argv[0]);
    /* abort MPI and terminate reporting failure */
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }
}

/* divide problem's matrix so that it's parts can be processed in parallel as efficiently as possible */
int divideWorkload(int rows, int columns, int workers) {
  int perimeter, r, c, best = 0;          // no "good" sub-matrices so far
  int min_perimeter = rows + columns + 1; // because, the "minimum" possible division is to 4 sub-matrices, each with perimeter = rows + columns
  for (r = 1; r <= workers; ++r) {
    if ((workers % r != 0) || (rows % r != 0)) continue;
    c = workers / r;
    if (columns % c != 0) continue;
    perimeter = rows / r + columns / c;
    if (perimeter < min_perimeter) {
      min_perimeter = perimeter;
      best = r;
    }
  }

  return best;  // best possible row's divisor
}

/* read matrix's initial state from a text-file */
void readInitialState(char* filename, char* matrix, int starting_row, int starting_column, int proc_rows, int proc_columns, int rows, int columns) {
  FILE *fd;

  if (filename)
    if ((fd = fopen(filename, "r")) == NULL) {
      perror("Error: Failed to open file for input");
      /* abort MPI and terminate reporting failure */
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      exit(EXIT_FAILURE);
    }

  int i,j;
  for (i = 1; i <= proc_rows; i++)
    for (j = 1; j <= proc_columns; j++)
      matrix[(proc_columns+2) * i + j] = 0;

  if (filename) {
    while (fscanf(fd, "%d %d\n", &i, &j) != EOF) {
      // printf("\t\t\tREAD %d %d\n\n\n", i, j);
      if ((i > starting_row) && (i <= starting_row + proc_rows) && (j > starting_column) && (j <= starting_column + proc_columns)) {
        matrix[(proc_columns+2) * i + j] = 1;
        // printf("OK!!! %d\n", matrix[(proc_columns+2) * i + j]);
      } else if (i > rows || j > columns) {
        fprintf(stderr, "Error: Bad input data at file %s", filename);
        /* abort MPI and terminate reporting failure */
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        exit(EXIT_FAILURE); 
      }
    }
    fclose(fd);
  } else {
    perror("Error: Failed to open file for input");
    /* abort MPI and terminate reporting failure */
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }
}

/* set a random initial state for the matrix */
void setRandomInitialState(char* matrix, int proc_rows, int proc_columns) {
  int i, j;
  for(i = 1; i <= proc_rows; i++)   //first we create an empty matrix
    for(j = 1; j <= proc_columns; j++)
        matrix[(proc_columns+2) * i + j] = DEAD;

  int num_of_organisms = rand() % (proc_rows + proc_columns) + 1;
  printf("initial number of organisms = %d\n", num_of_organisms);

  int counter = 0;
  for (counter = 0; counter < num_of_organisms; counter++) { //for every cell that we have to place in the matrix
    do {
        i = rand() % proc_rows + 1;
        j = rand() % proc_columns + 1;
    } while (matrix[(proc_columns+2)* i + j] == ALIVE);  // while the position that was randomly chosen is already occupied, randomly chose another one
    matrix[(proc_columns+2) * i + j] = ALIVE;      // an organism is placed
  }
}

/* "play" the game of life for a specific part of a matrix -- compute the next generation */
int nextGeneration(char* before, char* after, int first_row, int last_row, int first_column, int last_column, int columns) {
  int i, j, changed = 0;
#pragma omp parallel for shared(gen1, gen2) schedule(static) collapse(3)
  for (i = first_row; i <= last_row; i++) {
    for (j = first_column; j <= last_column; j++) {
      int neighbors = before[columns*(i-1)+j] + before[columns*i+(j-1)] + before[columns*(i+1)+j] + before[columns*i+(j+1)]
                              + before[columns*(i-1)+(j+1)] + before[columns*(i+1)+(j-1)] + before[columns*(i-1)+(j-1)] + before[columns*(i+1)+(j+1)];
      
      if (before[columns*i+j] == ALIVE) {     // if there is an organism
        if (neighbors <= 1) after[columns*i+j] = DEAD;  // dies from loneliness
        else if (neighbors <= 3) after[columns*i+j] = ALIVE; // lives
        else after[columns*i+j] = DEAD;   // dies from overpopulation
      } else {               // if there is no organism
        if (neighbors == 3) after[columns*i+j] = ALIVE;  // a new organism is born
        else after[columns*i+j] = DEAD;   // no organism
      }

      if (before[columns*i+j] != after[columns*i+j]) changed++;
    }
  }

  return changed; // how many matrix points changed status
}

/* check if there is no organism left alive */
char noneAlive(char* matrix, int proc_rows, int proc_columns) {
  int i, j;
  for (i = 1; i <= proc_rows; i++)
    for (j = 1; j <= proc_columns; j++)
      if (matrix[(proc_columns+2) * i + j] == ALIVE)  // if there is at least one living organism, then the simulation is not over
        return 0;

  return 1;   // if we reached this far, then the simulation is definitelly over (no organism left)
}

/* check for the same matrix for 2 consecutive generations */
char sameGenerations(char* gen1, char* gen2, int proc_rows, int proc_columns) {
  int i, j;
#pragma omp parallel for shared(gen1, gen2) schedule(static) collapse(3)
  for (i = 1; i <= proc_rows; i++)
    for (j = 1; j <= proc_columns; j++)
      if (gen1[(proc_columns+2) * i + j] != gen2[(proc_columns+2) * i + j]) // we definitelly don't have the same generations
        return 0;

  return 1;   // if we reached this far, we definitelly have the same generations
}

/* Get pointer to internal matrix position -- utility */
char *locate(char *matrix, int row, int column, int columns) {
  return &matrix[columns * row + column];
}

/* print the sub-matrix and the process that it belongs to -- debugging */
void printSubmatrix(int my_rank, char* matrix, int proc_rows, int proc_columns) {
  printf("my_rank = %d\n", my_rank);
  int i, j;
  for (i = 1; i <= proc_rows; i++) {
    for (j = 1; j <= proc_columns; j++)
      printf("%d ", matrix[(proc_columns+2) * i + j]);
    printf("\n");
  }
}