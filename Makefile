NVCC        = nvcc
ifeq (,$(shell which nvprof))
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include 
else
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include --std=c++03
endif
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 
EXE	        = reduction
OBJ	        = main.o support.o

default: naive optimized


main-optimized.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) -DOPTIMIZED

main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS) 

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

naive: $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

optimized: main-optimized.o support.o
	$(NVCC) main-optimized.o support.o -o $(EXE)-optimized $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE) $(EXE)-optimized
