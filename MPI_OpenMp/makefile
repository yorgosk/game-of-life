CC = mpicc
LIBS = -g -Wall -openmp
EXEC = gameOfLife_mpi_openmp
OBJECTS = main_mpi_openmp.o functions_mpi_openmp.o
SOURCES = main_mpi_openmp.c functions_mpi_openmp.c
HEADERS = header_mpi_openmp.h

gameOfLife_mpi_openmp: $(OBJECTS)
	$(CC) $(LIBS) -o $(EXEC) $(OBJECTS)

main_mpi_openmp.o: main_mpi_openmp.c
	$(CC) -c main_mpi_openmp.c

functions_mpi_openmp.o: functions_mpi_openmp.c
	$(CC) -c functions_mpi_openmp.c

.PHONY: clean

clean:
	rm -f $(EXEC) $(OBJECTS)

count:
	wc $(SOURCES) $(HEADERS)
