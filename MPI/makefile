CC = mpicc
LIBS = -g -Wall
EXEC = gameOfLife_mpi
OBJECTS = main_mpi.o functions_mpi.o
SOURCES = main_mpi.c functions_mpi.c
HEADERS = header_mpi.h

gameOfLife_mpi: $(OBJECTS)
	$(CC) $(LIBS) -o $(EXEC) $(OBJECTS)

main_mpi.o: main_mpi.c
	$(CC) -c main_mpi.c

functions_mpi.o: functions_mpi.c
	$(CC) -c functions_mpi.c

.PHONY: clean

clean:
	rm -f $(EXEC) $(OBJECTS)

count:
	wc $(SOURCES) $(HEADERS)
