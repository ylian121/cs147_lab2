/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

__device__ unsigned int warpDistribution[33] = {0};

__device__ void countWarpDistribution(){

      unsigned int mask = __popc(__activemask());
      if(threadIdx.x % 32 == 0){
        atomicAdd(&warpDistribution[mask],1);
      }
}

__device__ void printWarpDistribution(){
    printf("\n Warp Distribution: \n");
    for(int i = 0; i < 33; i++){
        printf("W%d: %u, ",i,warpDistribution[i]);
    }
    printf("\n\n");
}

__global__ void naiveReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // NAIVE REDUCTION IMPLEMENTATION
    __shared__ float partialSum[2*BLOCK_SIZE];
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    
    partialSum[t] = in[start + t];
    partialSum[blockDim.x+t] = in[start + blockDim.x+t];
    
    for(unsigned int stride = 1;
      stride <= blockDim.x;  stride *= 2)
    {
      
      __syncthreads();
      if (t % stride == 0)
        partialSum[2*t] += partialSum[2*t+stride];
    }
      
    if(threadIdx.x == 0 && blockIdx.x == 0)
      printWarpDistribution();  
}

__global__ void optimizedReduction(float *out, float *in, unsigned size)
{
    /********************************************************************
    Load a segment of the input vector into shared memory
    Traverse the reduction tree
    Write the computed sum to the output vector at the correct index
    ********************************************************************/

    // INSERT KERNEL CODE HERE
    // OPTIMIZED REDUCTION IMPLEMENTATION
    __shared__ float partialSum[2*BLOCK_SIZE];
    
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;
    
    partialSum[t] = in[start + t];
    partialSum[blockDim.x+t] = in[start + blockDim.x+t];
    
    for(unsigned int stride = 1;
      stride <= blockDim.x;  stride *= 2)
    {
      
      __syncthreads();
      if (t % stride == 0)
        partialSum[2*t] += partialSum[2*t+stride];
    }

    if(threadIdx.x == 0 && blockIdx.x == 0)
      printWarpDistribution();  
}
