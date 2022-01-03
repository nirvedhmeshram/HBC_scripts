NVCC=nvcc
ARCHFLAG = -arch=sm_37 #This is for k80, you have to change this for the GPU you use
all: cumsum
cumsum : cumsum.cu
	$(NVCC) $(ARCHFLAG) cumsum.cu -o cumsum
