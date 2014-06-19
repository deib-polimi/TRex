//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Daniele Rogora
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

__device__ Attribute computeVal(int i, int id, int offset, EventInfoSet genEvent, int size, int attrId, GPUParameter *param, parPasser paramSize, uint64_t win, bool &ok) {
    Attribute val;
    if (i>=size) {
        ok=false;
	val.type=INT;
        return val;
    }
    else {
        val = stack[id + offset].attr[attrId];
#pragma unroll
        for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
            int pp=paramSize.paramNum[p];
            if (pp==-1) break;
            ok *= d_checkComplexParameterForAggregate(id, offset, param, &genEvent);
        }
        return val;
    }
}

template <unsigned int blockSize> __global__ void reduceFirstMax(passer stackpage, EventInfoSet genEvent, int size, int minId, int state, int attrId, float *g_odata, GPUParameter *param, parPasser paramSize, int win, unsigned int n, bool isMax, int *res) {
    //sdata can't be blockSize because thread blockSize - 1 would go out of bounds in the second phase
    __shared__ float sdata[blockSize+32];
    __shared__ int counter;
    bool ok1 = true;
    bool ok2 = true;
    int ok=0;

    // perform first level of reduction, reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    if (tid==0) counter=0;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i<n) {
        unsigned int id = (minId+i)%ALLOC_SIZE;
        unsigned int nextI = i+blockSize;
        unsigned int nextId = (minId+nextI)%ALLOC_SIZE;

        int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;

        id %= PAGE_SIZE;


        int nextOffset = stackpage.pageIdx[nextId/PAGE_SIZE] * PAGE_SIZE;

        nextId %= PAGE_SIZE;


        Attribute at1 = computeVal(i, id, offset, genEvent, size, attrId, param, paramSize, win, ok1);
        Attribute at2 = computeVal(nextI, nextId, nextOffset, genEvent, size, attrId, param, paramSize, win, ok2);
        ok = ok1+ok2;
        __syncthreads();
        atomicAdd(&counter, ok);
        float val1, val2;
        if (at1.type==FLOAT) val1 = at1.floatVal;
        else val1 = (float) at1.intVal;
        if (at2.type==FLOAT) val2 = at2.floatVal;
        else val2 = (float) at2.intVal;

        //FIXME: it's not ok to simply set to 0 if i or nextI is out of bounds!
        //Is this enough?
        if (isMax) sdata[tid] = fmax(val1 * ok1 + SMALL_NUM * (1-ok1), val2 * ok2 + SMALL_NUM * (1-ok2));
        else sdata[tid] = fmin(val1 * ok1 + BIG_NUM * (1-ok1), val2 * ok2 + BIG_NUM * (1-ok2));

        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize>=512) {
        if (tid<256) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+256]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+256]);
        }
        __syncthreads();
    }
    if (blockSize>=256) {
        if (tid<128) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+128]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+128]);
        }
        __syncthreads();
    }

    if (blockSize>=128) {
        if (tid<64) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+64]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+64]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockSize>=64) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+32]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+32]);
            __syncthreads();
        }
        if (blockSize>=32) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+16]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+16]);
            __syncthreads();
        }
        if (blockSize>=16) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+8]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+8]);
            __syncthreads();
        }
        if (blockSize>= 8) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+4]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+4]);
            __syncthreads();
        }
        if (blockSize>= 4) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+2]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+2]);
            __syncthreads();
        }
        if (blockSize>= 2) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+1]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+1]);
        }
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
        atomicAdd(res, counter);
    }
}

template <unsigned int blockSize> __global__ void reduceMax(float *g_idata, float *g_odata, unsigned int n, bool isMax) {
    __shared__ float sdata[blockSize+32];
    // perform first level of reduction, reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i<n) {
        if (isMax) sdata[tid] = fmax(g_idata[i], g_idata[i+blockSize]);
        else sdata[tid] = fmin(g_idata[i], g_idata[i+blockSize]);
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize>=512) {
        if (tid<256) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+256]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+256]);
        }
        __syncthreads();
    }
    if (blockSize>=256) {
        if (tid<128) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+128]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+128]);
        }
        __syncthreads();
    }

    if (blockSize>=128) {
        if (tid<64) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+64]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+64]);
        }
        __syncthreads();
    }

    if (tid < 32) {
        if (blockSize>=64) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+32]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+32]);
            __syncthreads();
        }
        if (blockSize>=32) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+16]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+16]);
            __syncthreads();
        }
        if (blockSize>=16) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+8]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+8]);
            __syncthreads();
        }
        if (blockSize>= 8) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+4]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+4]);
            __syncthreads();
        }
        if (blockSize>= 4) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+2]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+2]);
            __syncthreads();
        }
        if (blockSize>= 2) {
            if (isMax) sdata[tid] = fmax(sdata[tid],sdata[tid+1]); //sdata[tid] += sdata[tid+256];
            else sdata[tid] = fmin(sdata[tid],sdata[tid+1]);
        }
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize> __global__ void reduceFirstSumFloat(passer stackpage, EventInfoSet genEvent, int size, int minId, int state, int attrId, float *g_odata, GPUParameter *param, parPasser paramSize, int win, unsigned int n, int *res, AggregateFun fun) {
    __shared__ volatile float sdata[blockSize+32];
    float val;
    __shared__ int counter;
    bool ok1 = true;
    bool ok2 = true;
    int ok;

    // perform first level of reduction, reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    if (tid==0) counter=0;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i<n) {
        unsigned int id = (minId+i)%ALLOC_SIZE;
        unsigned int nextI = i+blockSize;
        unsigned int nextId = (minId+nextI)%ALLOC_SIZE;

        int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;

        id %= PAGE_SIZE;


        int nextOffset = stackpage.pageIdx[nextId/PAGE_SIZE] * PAGE_SIZE;

        nextId %= PAGE_SIZE;


        Attribute at1 = computeVal(i, id, offset, genEvent, size, attrId, param, paramSize, win, ok1);
        Attribute at2 = computeVal(nextI, nextId, nextOffset, genEvent, size, attrId, param, paramSize, win, ok2);
        ok = ok1+ok2;
        __syncthreads();
        if (fun!=SUM) atomicAdd(&counter, ok);
        val=0;
        if (at1.type == FLOAT) val = at1.floatVal * ok1;
        else val += (float) at1.intVal * ok1;

        if (at2.type == FLOAT) val += at2.floatVal * ok2;
        else val += (float) at2.intVal * ok2;

        sdata[tid] += val;
        i += gridSize;
    }
    if (fun!=COUNT) {
        __syncthreads();
        // do reduction in shared mem
        if (blockSize>=512) {
            if (tid<256) sdata[tid] += sdata[tid+256];
            __syncthreads();
        }
        if (blockSize>=256) {
            if (tid<128) sdata[tid] += sdata[tid+128];
            __syncthreads();
        }

        if (blockSize>=128) {
            if (tid<64) sdata[tid] += sdata[tid+64];
            __syncthreads();
        }

        if (tid < 32) {
            if (blockSize>=64) sdata[tid] += sdata[tid+32];
            __syncthreads();
            if (blockSize>=32) sdata[tid] += sdata[tid+16];
            __syncthreads();
            if (blockSize>=16) sdata[tid] += sdata[tid+ 8];
            __syncthreads();
            if (blockSize>= 8) sdata[tid] += sdata[tid+ 4];
            __syncthreads();
            if (blockSize>= 4) sdata[tid] += sdata[tid+ 2];
            __syncthreads();
            if (blockSize>= 2) sdata[tid] += sdata[tid+ 1];
        }
    }
    else __syncthreads();
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
        atomicAdd(res, counter);
    }
}


template <unsigned int blockSize> __global__ void reduceSumFloat(float *g_idata, float *g_odata, unsigned int n) {
    __shared__ float sdata[blockSize+32];
    // perform first level of reduction, reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridSize).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i<n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize>=512) {
        if (tid<256) sdata[tid] += sdata[tid+256];
        __syncthreads();
    }
    if (blockSize>=256) {
        if (tid<128) sdata[tid] += sdata[tid+128];
        __syncthreads();
    }
    if (blockSize>=128) {
        if (tid<64) sdata[tid] += sdata[tid+64];
        __syncthreads();
    }
    if (tid < 32) {
        if (blockSize>=64) sdata[tid] += sdata[tid+32];
        __syncthreads();
        if (blockSize>=32) sdata[tid] += sdata[tid+16];
        __syncthreads();
        if (blockSize>=16) sdata[tid] += sdata[tid+ 8];
        __syncthreads();
        if (blockSize>= 8) sdata[tid] += sdata[tid+ 4];
        __syncthreads();
        if (blockSize>= 4) sdata[tid] += sdata[tid+ 2];
        __syncthreads();
        if (blockSize>= 2) sdata[tid] += sdata[tid+ 1];
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}