/* File: main_cuda.c
   Authors: Kamaras Georgios
   Date:
*/
#include "header_cuda.h"

int main(int argc, char** argv) {
  	char *initial = NULL;
	int rows, columns;
	char *filename = NULL;
	/* Use provided arguments */
	/* Initialize problem's representation */
	initializeRepresentation(argc, argv, &filename, &rows, &columns);
  	/* Initialize simulation's arrays */
	initial = malloc(rows*columns*sizeof(char));
	/* if memory allocation failed, report the issue */
	if (!initial) {
		fprintf(stderr, "%s: Error: malloc failed\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	/* check to see if we have a random initial state */
	if (argc == 1)	/* random initial state */
		setRandomInitialState(initial, rows, columns);
	else 	/* specific initial state from text-file */
		readInitialState(filename, initial, rows, columns);

#ifdef _DEBUGGING_
	/* for debugging */
	printMatrix(initial, rows, columns);
#endif

	/* Go to our simulation -- compute the next generations until you reach an end-condition and return the elapsed (execution) time */
  	float timer = playTheGame(initial, rows, columns);

#ifdef _DEBUGGING_
  	/* for debugging */
	printMatrix(initial, rows, columns);
#endif
  	/* Print elapsed (execution) time, converted in seconds */
  	printf("Total execution time: %3.1f msecs\n", timer);

  	/* Free dynamically allocated memory */
	free(initial);

  	exit(EXIT_SUCCESS);
}
