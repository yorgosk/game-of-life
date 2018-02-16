CC = gcc
CUDAC = nvcc
CFLAGS = -O2 -lm
EXEC = gameOfLife_cuda
OBJECTS = main_cuda.o functions_cuda.o gameOfLife_cuda.o
SOURCES = main_cuda.c functions_cuda.c gameOfLife_cuda.cu
HEADERS = header_cuda.h

gameOfLife_cuda: $(OBJECTS)
	$(CUDAC) -o $(EXEC) $(OBJECTS) $(CFLAGS)

main_cuda.o: main_cuda.c
	$(CC) -c main_cuda.c $(CFLAGS)

functions_cuda.o: functions_cuda.c
	$(CC) -c functions_cuda.c $(CFLAGS)

gameOfLife_cuda.o: gameOfLife_cuda.cu
	$(CUDAC) -c gameOfLife_cuda.cu

.PHONY: clean

clean:
	rm -f $(EXEC) $(OBJECTS)

count:
	wc $(SOURCES) $(HEADERS)
