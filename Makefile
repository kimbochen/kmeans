CC=gcc
CFLAGS=-Wall -O2
CUTLASS_LIB=-I/nfshome/tchen307/cutlass/include -I/nfshome/tchen307/cutlass/tools/util/include


cutlass:
	nvcc $(CUTLASS_LIB) sqrsum.cu -o sqrsum


seq: seq_kmeans.o file_io.o
	$(CC) $(CFLAGS) seq_kmeans.o file_io.o seq_main.c -o seq_km

seq_kmeans.o: seq_kmeans.c
	$(CC) $(CFLAGS) -c seq_kmeans.c

file_io.o: file_io.c
	$(CC) $(CFLAGS) -c file_io.c


test1: seq_km
	./seq_km tcases/tcase1.txt tcases/tcase1_cfg.txt

test2: seq_km
	./seq_km tcases/tcase2.txt tcases/tcase2_cfg.txt


clean:
	rm -f sqrsum *.o seq_km
