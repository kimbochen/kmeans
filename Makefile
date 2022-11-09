CC=gcc
CFLAGS=-Wall -O2
CUTLASS_LIB=-I/nfshome/tchen307/cutlass/include -I/nfshome/tchen307/cutlass/tools/util/include
CUBLAS_LIB=-lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
EXTRA=


cutlass:
	nvcc $(CUTLASS_LIB) sqrsum.cu -o sqrsum


seq: seq_kmeans.o file_io.o
	$(CC) $(CFLAGS) $(EXTRA) seq_kmeans.o file_io.o seq_main.c -o seq_km

seq_kmeans.o: seq_kmeans.c
	$(CC) $(CFLAGS) -c seq_kmeans.c

file_io.o: file_io.c
	$(CC) $(CFLAGS) -c file_io.c


cuda: cuda_main.cu cuda_kmeans.cu
	nvcc $(CUBLAS_LIB) cuda_main.cu cuda_kmeans.cu


ID=
test: seq_km
	./seq_km tcases/tcase$(ID).txt tcases/tcase$(ID)_cfg.txt
	./a.out tcases/tcase$(ID).txt tcases/tcase$(ID)_cfg.txt

clean:
	rm -f sqrsum *.o seq_km a.out
