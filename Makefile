CC=nvcc
ARCH=-arch=sm_86
CUBLAS_LIB=-lcublas_static -lculibos -lcudart_static -lpthread -ldl -lcublasLt_static
EXT=


all: km_cuda km_cublas km_tcu


km_cuda: main.cu cuda_kmeans.cu
	$(CC) $(ARCH) $(CUBLAS_LIB) -DKERNEL=1 $(EXT) $^ -o $@

km_cublas: main.cu cublas_kmeans.cu
	$(CC) $(ARCH) $(CUBLAS_LIB) -DKERNEL=2 $(EXT) $^ -o $@

km_tcu: main.cu tcu_kmeans.cu
	$(CC) $(ARCH) $(CUBLAS_LIB) -DKERNEL=3 $(EXT) $^ -o $@


clean:
	rm -f km_*
