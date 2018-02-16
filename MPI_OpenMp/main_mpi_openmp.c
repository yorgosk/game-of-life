/* File: main_mpi_openmp.c
   Author: Kamaras Georgios
   Date: 8/10/2017
*/
#include "header_mpi_openmp.h"

int main(int argc, char** argv) {
	/* Our problem's representation variables */
	int rows, columns, rows_div, columns_div;
	char *filename = NULL;
	/* For time calculation */
	double timer;
	/* repetitions counter */
	int reps = 0;

	/* MPI communicator, topology and number of processes in the communicator */
	MPI_Comm comm;
	int my_rank, comm_sz;

	/* Initialize MPI (tells MPI to do all the necessary setup) */
	MPI_Init(&argc, &argv);

	/* Get current process rank and number of processes */
  	comm = MPI_COMM_WORLD;
  	MPI_Comm_size(comm, &comm_sz);
  	MPI_Comm_rank(comm, &my_rank);
	/* MPI status, data-types and requests */
	MPI_Status status;											// status
	MPI_Datatype row_type, column_type;							// data-types
	MPI_Request send_north, send_east, send_south, send_west, 	// "send"-kind requests
				send_north_east, send_north_west, send_south_east, send_south_west,
				recv_north, recv_east, recv_south, recv_west,	// "receive"-kind requests
				recv_north_east, recv_north_west, recv_south_east, recv_south_west;
	// MPI_Request send_end, recv_end, send_over, recv_over;
	/* Neighbors' processes numbers */
	int north, east, south, west, north_east, north_west, south_east, south_west;
	/* For termination criteria */
	int sameFlag = 0, endFlag = 0, localEndFlag = 0, changed = 0;

	/* Use provided arguments */
	/* Initialize problem's representation */
	initializeRepresentation(argc, argv, &filename, &rows, &columns);
	if (my_rank == 0) {
		/* divide work so that it can be distributed to 'communicator's processes in an optimal way */
		rows_div = divideWorkload(rows, columns, comm_sz);
		columns_div = comm_sz / rows_div;
		/* if (optimal) division is impossible */
		if ((rows_div <= 0) || (rows % rows_div != 0) || (comm_sz % rows_div != 0) || (columns % columns_div != 0)) {
			/* report the issue */
			fprintf(stderr, "%s: Error: Cannot divide to processes\n", argv[0]);
			/* abort MPI and terminate reporting failure */
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			exit(EXIT_FAILURE);
		}
	}

	/* Broadcast parameters */
	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&rows_div, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&columns_div, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Compute number of rows and columns per process */
	int proc_rows = rows / rows_div;
	int proc_columns = columns / columns_div;

	/* Create column's and row's data-types */
	MPI_Type_vector(proc_rows, 1, proc_columns+2, MPI_CHAR, &column_type);
	MPI_Type_commit(&column_type);
	MPI_Type_contiguous(proc_columns, MPI_CHAR, &row_type);
	MPI_Type_commit(&row_type);
	/* Corner's data type */
	char corner_type;

	/* Compute starting row and column */
	int starting_row = (my_rank / columns_div) * proc_rows;
	int starting_column = (my_rank % columns_div) * proc_columns;

	char *before = NULL, *after = NULL, *temp = NULL, *stepsTemp = NULL;
	/* Initialize simulation's arrays */
	/* calloc() because it initializes the buffer */
	/* source: https://stackoverflow.com/questions/1538420/difference-between-malloc-and-calloc */
	before = calloc((proc_rows+2) * (proc_columns+2), sizeof(char));
	after = calloc((proc_rows+2) * (proc_columns+2), sizeof(char));
	stepsTemp = calloc((proc_rows+2) * (proc_columns+2), sizeof(char));
	/* if memory allocation failed */
	if (!before || !after || !stepsTemp) {
		/* report the issue */
		fprintf(stderr, "%s: Error: malloc failed\n", argv[0]);
		/* abort MPI and terminate reporting failure */
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUGGING_
	/* debugging */
	printf("\nmy_rank = %d rows = %d columns %d rows_div = %d columns_div = %d\nproc_rows = %d proc_columns = %d starting_row = %d starting_column = %d\n\n", 
		      my_rank, rows, columns, rows_div, columns_div, proc_rows, proc_columns, starting_row, starting_column);
#endif
	/* check to see if we have a random initial state */
	if (argc == 1) {	/* random initial state, various things can be done if no arguments, I have left uncommented what works the best */
		setRandomInitialState(before, proc_rows, proc_columns);
		// fprintf(stderr, "Usage: mpiexec -n <np> ./gameOfLife_mpi -f <filepath> -r <no.rows> -c <no.columns> .\nPlease, take a look at run.sh for simple usage case.\n");
	 //    exit(EXIT_FAILURE);
	} else 	/* specific initial state from text-file */
		readInitialState(filename, before, starting_row, starting_column, proc_rows, proc_columns, rows, columns);

#ifdef _NGENERATIONS_
	int i, j;
	for (i = 0; i < proc_rows; i++) for (j = 0; j < proc_columns; j++) stepsTemp[i*proc_columns + j] = before[i*proc_columns + j];
#endif

	/* Compute neighbors */ 				/*	0 1 2
												3 4 5
												6 7 8	*/
	if (starting_row)								// northern neighbor
		north = my_rank - columns_div;
	else
		north = rows_div * columns_div - (columns_div - my_rank);
	if (starting_column + proc_columns != columns)	// eastern neighbor
		east = my_rank + 1;
	else
		east = my_rank - columns_div + 1;
	if (starting_row + proc_rows != rows)			// southern neighbor
		south = my_rank + columns_div;
	else
		south = my_rank % columns_div;
	if (starting_column)							// western neighbor
		west = my_rank - 1;
	else
		west = my_rank + columns_div - 1;
	if ((starting_row && (starting_column + proc_columns != columns))
		|| !starting_row && (starting_column + proc_columns != columns))		// north-eastern neighbor
		north_east = north + 1;
	else if (starting_row && !(starting_column + proc_columns != columns))
		north_east = east - columns_div;
	else
		north_east = columns_div * rows_div - columns_div;
	if ((starting_row && starting_column)
		|| !starting_row && starting_column)									// north-western neighbor
		north_west = north - 1;
	else if (starting_row && !starting_column)
		north_west =  west - columns_div;
	else
		north_west = columns_div * rows_div - 1;
	if ((starting_row + proc_rows != rows) && (starting_column + proc_columns != columns)
		|| !(starting_row + proc_rows != rows) && (starting_column + proc_columns != columns))	// south-eastern neighbor
		south_east = south + 1;
	else if ((starting_row + proc_rows != rows) && !(starting_column + proc_columns != columns))
		south_east = east + columns_div;
	else
		south_east = 0;
	if (((starting_row + proc_rows != rows) && starting_column)
		|| (!(starting_row + proc_rows != rows) && starting_column))							// south-western neighbor
		south_west = south - 1;
	else if ((starting_row + proc_rows != rows) && !starting_column)
		south_west = west + columns_div;
	else
		south_west = columns_div - 1;
#ifdef _DEBUGGING_
	/* debugging */
	printf("\t\nmy_rank = %d\n\tnorth = %d east = %d south = %d west %d\n\tnorth-west = %d north-east = %d south-east = %d south-west = %d\n\n",
				my_rank, north, east, south, west, north_west, north_east, south_east, south_west);
#endif
	/* Ensures that no process will return from calling it
		until every process in the communicator has started calling it */
	MPI_Barrier(MPI_COMM_WORLD);

	/* Initialize timer's value */
	timer = MPI_Wtime();
#ifdef _DEBUGGING_
	/* debugging */
	printf("printing before\n");
	printSubmatrix(my_rank, before, proc_rows, proc_columns);
#endif
	/* Our simulation's loop */
	temp = before;
// #ifdef _DEBUGGING_
// 	while (!endFlag && reps < MAXREPS) {
// #endif
	while (!endFlag) {
		sameFlag = 0;
		endFlag = 1;
		localEndFlag = 0;
		changed = 0;

		/* send and receive submatrixes borders from neighbors' processes */
		/* send to the northern neighbor and receive from him */
		MPI_Isend(locate(before, 1, 1, proc_columns+2), 1, row_type, north, 0, MPI_COMM_WORLD, &send_north);
		MPI_Irecv(locate(before, 0, 1, proc_columns+2), 1, row_type, north, 0, MPI_COMM_WORLD, &recv_north);
		/* send to the eastern neighbor and receive from him */
		MPI_Isend(locate(before, 1, proc_columns, proc_columns+2), 1, column_type, east, 0, MPI_COMM_WORLD, &send_east);
		MPI_Irecv(locate(before, 1, proc_columns+1, proc_columns+2), 1, column_type, east, 0, MPI_COMM_WORLD, &recv_east);
		/* send to the southern neighbor and receive from him */
		MPI_Isend(locate(before, proc_rows, 1, proc_columns+2), 1, row_type, south, 0, MPI_COMM_WORLD, &send_south);
		MPI_Irecv(locate(before, proc_rows+1, 1, proc_columns+2), 1, row_type, south, 0, MPI_COMM_WORLD, &recv_south);
		/* send to the western neighbor and receive from him */
		MPI_Isend(locate(before, 1, 1, proc_columns+2), 1, column_type, west, 0, MPI_COMM_WORLD, &send_west);
		MPI_Irecv(locate(before, 1, 0, proc_columns+2), 1, column_type, west, 0, MPI_COMM_WORLD, &recv_west);
		/* send to north-western neighbor and receive from him */
		MPI_Isend(locate(before, 1, 1, proc_columns+2), 1, MPI_CHAR, north_west, 0, MPI_COMM_WORLD, &send_north_west);
		MPI_Irecv(locate(before, 0, 0, proc_columns+2), 1, MPI_CHAR, north_west, 0, MPI_COMM_WORLD, &recv_north_west);
		/* send to north-eastern neighbor and receive from him */
		MPI_Isend(locate(before, 1, proc_columns, proc_columns+2), 1, MPI_CHAR, north_east, 0, MPI_COMM_WORLD, &send_north_east);
		MPI_Irecv(locate(before, 0, proc_columns+1, proc_columns+2), 1, MPI_CHAR, north_east, 0, MPI_COMM_WORLD, &recv_north_east);
		/* send to south-eastern neighbor and receive from him */
		MPI_Isend(locate(before, proc_rows, proc_columns, proc_columns+2), 1, MPI_CHAR, south_east, 0, MPI_COMM_WORLD, &send_south_east);
		MPI_Irecv(locate(before, proc_rows+1, proc_columns+1, proc_columns+2), 1, MPI_CHAR, south_east, 0, MPI_COMM_WORLD, &recv_south_east);
		/* send to sourh-western neighbor and receive from him */
		MPI_Isend(locate(before, proc_rows, 1, proc_columns+2), 1, MPI_CHAR, south_west, 0, MPI_COMM_WORLD, &send_south_west);
		MPI_Irecv(locate(before, proc_rows+1, 0, proc_columns+2), 1, MPI_CHAR, south_west, 0, MPI_COMM_WORLD, &recv_south_west);

		/* "play" the game of life for a given submatrix -- compute the next generation -- for the inner data */
		changed += nextGeneration(before, after, 1, proc_rows, 1, proc_columns, proc_columns+2);

		/* for each border, wait until it has been received and then use it to compute the next generation for the inner data */
		/* northern border */
		MPI_Wait(&recv_north, &status);
		/* "play" the game of life for a given submatrix -- compute the next generation -- for the border data */
		changed += nextGeneration(before, after, 1, 1, 2, proc_columns-1, proc_columns+2);
		/* eastern border */
		MPI_Wait(&recv_east, &status);
		/* "play" the game of life for a given submatrix -- compute the next generation -- for the border data */
		changed += nextGeneration(before, after, 2, proc_rows-1, proc_columns, proc_columns, proc_columns+2);
		/* southern border */
		MPI_Wait(&recv_south, &status);
		/* "play" the game of life for a given submatrix -- compute the next generation -- for the border data */
		changed += nextGeneration(before, after, proc_rows, proc_rows, 2, proc_columns-1, proc_columns+2);
		/* western border */
		MPI_Wait(&recv_west, &status);
		/* "play" the game of life for a given submatrix -- compute the next generation -- for the border data */
		changed += nextGeneration(before, after, 2, proc_rows-1, 1, 1, proc_columns+2);

		/* for each corner, wait until it has been received and then use it to compute the next generation for the corner data */
		/* north-western border */
		MPI_Wait(&recv_north_west, &status);
		/* top-left (north-western) border */
		changed += nextGeneration(before, after, 1, 1, 1, 1, proc_columns+2);
		/* north-eastern border */
		MPI_Wait(&recv_north_east, &status);
		/* top-right (north-eastern) border */
		changed += nextGeneration(before, after, 1, 1, proc_columns, proc_columns, proc_columns+2);
		/* south-eastern border */
		MPI_Wait(&recv_south_east, &status);
		/* bottom-right (south-eastern) border */
		changed += nextGeneration(before, after, proc_rows, proc_rows, proc_columns, proc_columns, proc_columns+2);
		/* south-western border */
		MPI_Wait(&recv_south_west, &status);
		/* bottom-left (south-western) border */
		changed += nextGeneration(before, after, proc_rows, proc_rows, 1, 1, proc_columns+2);

		/* wait for all borders to be sent */
		/* wait northern border */
		MPI_Wait(&send_north, &status);
		/* wait eastern border */
		MPI_Wait(&send_east, &status);
		/* wait southern border */
		MPI_Wait(&send_south, &status);
		/* wait western border */
		MPI_Wait(&send_west, &status);
		/* wait for all corners to be sent */
		/* wait north-eastern border */
		MPI_Wait(&send_north_east, &status);
		/* wait north-western border */
		MPI_Wait(&send_north_west, &status);
		/* wait south-western border */
		MPI_Wait(&send_south_west, &status);
		/* wait south-eastern border */
		MPI_Wait(&send_south_east, &status);

		/* swap simulation's matrixes */
		temp = before;
		before = after;
		after = temp;

		/* check for end-condition locally */
		sameFlag = (changed == 0);
		
		if (sameFlag) localEndFlag = 1;
		else localEndFlag = noneAlive(before, proc_rows, proc_columns);	// check for no-organisms

#ifdef _NGENERATIONS_
		if (!(reps%STEPS) && !localEndFlag) {
			localEndFlag = sameGenerations(before, stepsTemp, proc_rows, proc_columns);
			for (i = 0; i < proc_rows; i++) for (j = 0; j < proc_columns; j++) stepsTemp[i*proc_columns + j] = before[i*proc_columns + j];
		}
#endif

#ifdef _DEBUGGING_
		printf("\t\t\t%d: localEndFlag %d\n", my_rank, localEndFlag);
#endif

		/* gather the results of all local end-condition checks to the root process */
		int *sub_endFlags = NULL;
		if (my_rank == 0) sub_endFlags = malloc(sizeof(int) * comm_sz);
		MPI_Gather(&localEndFlag, 1, MPI_INT, sub_endFlags, 1, MPI_INT, 0, MPI_COMM_WORLD);

		/* compute the global end-contition on the root process */
		if (my_rank == 0) {
			int i;
			for (i = 0; i < comm_sz; i++) endFlag *= sub_endFlags[i];
		}
		/* broadcast the global end-condition */
		MPI_Bcast(&endFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		reps++;
	}
#ifdef _DEBUGGING_
	printf("\t\t\t\t%d OUT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", my_rank);
	/* debugging */
	if (reps == MAXREPS) printf("%d maximum number of repetitions reached!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", my_rank);
	printf("printing after\n");
	printSubmatrix(my_rank, before, proc_rows, proc_columns);
#endif
	/* Compute elapsed time */
	timer = MPI_Wtime() - timer;

	/* Get elapsed times from other processes and find the maximum */
	double remote_time;
	if (my_rank != 0)
		MPI_Send(&timer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	else {
		int i;
		for (i = 1; i < comm_sz; i++) {
			MPI_Recv(&remote_time, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			if (remote_time > timer) timer = remote_time;
		}
		printf("Best time overall: %f\n", timer);
	}

	/* Free dynamically allocated memory */
	free(before);
	before = NULL;
	free(after);
	after = NULL;
	free(stepsTemp);
	stepsTemp = NULL;
	MPI_Type_free(&column_type);
	MPI_Type_free(&row_type);

	/* Finalize MPI (tells MPI we’re done, so clean up anything allocated for this program) */
	MPI_Finalize();

	exit(EXIT_SUCCESS);
}
