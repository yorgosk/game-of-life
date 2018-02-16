/* File: functions_cuda.c
  Authors: Kamaras Georgios
  Date:
*/
#include "header_cuda.h"

/* Initialize problem's representation */
void initializeRepresentation(int argc, char** argv, char** filename, int* rows, int* columns) {
  /* Set-up based on command line's arguments */
  if (argc == 1) {  /* various things can be done if no arguments, I have left uncommented what works the best */
    // srand(time(NULL));
    // int possibleDimensions[6] = {2, 3, 4, 5, 6, 7};  // for 4, 9, 16, 25, 36 and 49 squares respectively
    // int randomIndex = rand() % 6;
    // *rows = possibleDimensions[randomIndex];
    // *columns = *rows;
    *rows = DEFAULT_ROWS;
    *columns = DEFAULT_COLUMNS;
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
        exit(EXIT_FAILURE);
      }
    }
    // printf("%s %d %d\n", *filename, *rows, *columns);
  } else {
    /* miss-program */
    fprintf(stderr, "%s: Error: Insufficient number of arguments\n", argv[0]);
    exit(EXIT_FAILURE);
  }
}

/* read matrix's initial state from a text-file */
void readInitialState(char* filename, char* matrix, int rows, int columns) {
  FILE *fd;
  int i, j;

  if (filename)
    if ((fd = fopen(filename, "r")) == NULL) {
      perror("Error: Failed to open file for input");
      exit(EXIT_FAILURE);
    }

  for (i = 0; i < rows; i++)
    for (j = 0; j < columns; j++)
      matrix[columns * i + j] = DEAD;

  if (filename) {
    while (fscanf(fd, "%d %d\n", &i, &j) != EOF) {
      // printf("%d %d\n", i, j);
      matrix[columns * (i-1) + j - 1] = ALIVE;
    }
    fclose(fd);
  } else {
    perror("Error: Failed to open file for input");
    exit(EXIT_FAILURE);
  }
}

/* set a random initial state for the matrix */
void setRandomInitialState(char* matrix, int rows, int columns) {
  int i, j;
  for(i = 0; i < rows; i++)   //first we create an empty matrix
    for(j = 0; j < columns; j++)
      matrix[columns * i + j] = DEAD;

  int num_of_cells = rand() % (rows + columns) + 1;
  int counter = 0;
  printf("initial number of cells = %d\n", num_of_cells);

  for (counter = 0; counter < num_of_cells; counter++) { //for every cell that we have to place in the matrix
    do {
      i = rand() % rows;
      j = rand() % columns;
    } while (matrix[columns * i + j]);  // while the position that was randomly chosen is already occupied, randomly chose another one
    matrix[columns * i + j] = ALIVE;      // an organism is placed
  }
}

/* print problem's matrix -- for debugging */
void printMatrix(char* matrix, int rows, int columns) {
  int i, j;
  printf("\nProblem's snapshot: rows = %d  columns = %d\n\n", rows, columns);
  for (i = 0; i < rows; i++) {
    for (j = 0; j < columns; j++) {
      printf("%d\t", matrix[columns * i + j]);
    }
    putchar('\n');
  }
}
