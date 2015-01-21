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


__global__ void computeComplexEventsSingle(passer stackpage, EventInfoSet *prevResults, EventInfoSet *currentResult, int *currentIndex, int state, int minId, int size, GPUParameter *param, parPasser paramSize, int sequenceLen, uint64_t win, int referredState, bool firstWithin) {
    __shared__ volatile int maxIdx;
    int i = threadIdx.x;
    int j = blockIdx.y;
    if (i == 0) {
        if (!firstWithin) {
            maxIdx = 0;
        }
        else {
            maxIdx = BIG_NUM;
        }
    }
    if (i>=size) return; // Out of range

    int id = (minId+i)%ALLOC_SIZE;
    int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;

    id %= PAGE_SIZE;

    uint64_t timestamp = stack[id + offset].timestamp;
    bool valid = (timestamp < prevResults[j].infos[referredState].timestamp && timestamp > prevResults[j].infos[referredState].timestamp-win);

    //This branch doesn't diverge
    if (paramSize.paramNum[0] != -1) {
#pragma unroll
        for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
            int pp=paramSize.paramNum[p];
            if (pp==-1) break;
 	    valid *= d_checkComplexParameter(id, offset, &param[pp], &prevResults[j], false, state);
        }
    }


    if (!valid) return;

    //This branch doesn't diverge
    if (!firstWithin) {
        int oldMax = atomicMax((int *)&maxIdx, i);
        if (oldMax > i) return; // Not the last one
        __syncthreads();
        if (i < maxIdx) return;
    }
    else {
        int oldMin = atomicMin((int *)&maxIdx, i);
        if (oldMin < i) return; // Not the first one
        __syncthreads();
        if (i > maxIdx) return;
    }
    __threadfence();

    int writeIndex = atomicAdd(currentIndex, 1);

    if (writeIndex >= MAX_GEN) return;

    currentResult[writeIndex].infos[state] = stack[id + offset];
#pragma unroll
    for (int k=sequenceLen-1; k>state; k--) {
        currentResult[writeIndex].infos[k] = prevResults[j].infos[k];
    }
}

#ifndef CHECK_NEG_PARALLEL
__device__ bool gcheckN(EventInfo *ev1, EventInfoSet prevResults, GPUNegation *negations, int negNumber, negsPasser negsPages, GPUParameter *param, int state) {
    int negsOffset, negsId, id;

    uint64_t negTimestamp, maxTS, minTS;
    for (int n=0; n<negNumber; n++) {
        for (int i=0; i<negsPages.size[n]; i++) {
            id = (negsPages.minId[n] + i)% ALLOC_SIZE;
            negsOffset = negsPages.pages[n][id/PAGE_SIZE] * PAGE_SIZE;
            negsId = id % PAGE_SIZE;
            negTimestamp = stack[negsId + negsOffset].timestamp;

            if (negations[n].lowerId < 0)	{
		if (negations[n].upperId == state) maxTS = ev1->timestamp;
		//This should happen only when checking the terminator
		else maxTS = prevResults.infos[negations[n].upperId].timestamp;
                minTS = maxTS - negations[n].lowerTime;
            } else {
		maxTS = prevResults.infos[negations[n].upperId].timestamp;
                minTS = ev1->timestamp; // partialResult->indexes[neg->lowerId]->getTimeStamp();
            }
            //maxTS and minTS negation events are not valid; Jan 2015
            if (negTimestamp < maxTS && negTimestamp > minTS) {
	      int lres=1;
		#pragma unroll
                for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
                    int pp=negsPages.parameters[n][p];
                    if (pp==-1) break;
// 		    if (param[pp].lastIndex < state) continue; //it requires info that we still don't have; go on, it's ok ATM
		    lres *= d_checkComplexParameterForNegation(ev1, negsId, negsOffset, &param[pp], &prevResults, true, state);
                }
                 if (lres==1) return false;
            }
        }
    }
    return true;
}
#endif

__global__ void computeComplexEventsSingleG(passer stackpage, EventInfoSet *prevResults, EventInfoSet *currentResult, int *currentIndex, volatile int *maxTS, uint8_t *alive, int state, int minId, int size, GPUParameter *param, parPasser paramSize, int sequenceLen, uint64_t win, int referredState, bool firstWithin, uint8_t aliveChecker, GPUNegation *negations, int negsNum, negsPasser negPages, GPUParameter *nParam) {
    __shared__ volatile int maxIdxlocal;
    if (threadIdx.x == 0) {
        if (!firstWithin) maxIdxlocal = 0;
        else maxIdxlocal = BIG_NUM;
    }
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y;

    int id = (minId+x)%ALLOC_SIZE;
    #ifdef CHECK_NEG_PARALLEL
    int aliveIdx = id + y*MAX_NEW_EVENTS;
#endif
    
    if (x>=size
      #ifdef CHECK_NEG_PARALLEL
      || alive[aliveIdx]==aliveChecker
#endif
    ) return;//  return; // Out of range
    int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;

    id %= PAGE_SIZE;

    uint64_t timestamp = stack[id + offset].timestamp;
    int valid = (timestamp < prevResults[y].infos[referredState].timestamp && timestamp > prevResults[y].infos[referredState].timestamp-win);

    if (paramSize.paramNum[0] != -1) {
#pragma unroll
        for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
            int pp=paramSize.paramNum[p];
            if (pp==-1) break;
            valid *= d_checkComplexParameter(id, offset, &param[pp], &prevResults[y], false, state);
        }
    }

    if (valid==0
      #ifndef CHECK_NEG_PARALLEL
      || gcheckN(&stack[id + offset], prevResults[y], negations, negsNum, negPages, nParam, state)==false
      #endif
	) {
        return;
    }
    __syncthreads();

    if (!firstWithin) {
        int oldMax = atomicMax((int *)&maxIdxlocal, x);
        if (x < maxIdxlocal) return;
        oldMax = atomicMax((int *)&(maxTS[y]), x);
	__threadfence();
        // it is less OR EQUAL because if 0 is the only valid thread it must survive

        if (maxTS[y] == x) {
            currentResult[y].infos[state] = stack[id + offset];
#pragma unroll
            for (int k=sequenceLen-1; k>state; k--) {
                currentResult[y].infos[k] = prevResults[y].infos[k];
            }
        }
    }
    else {
        int oldMin = atomicMin((int *)&maxIdxlocal, x);
        if (x > maxIdxlocal) return;
        oldMin = atomicMin((int *)&(maxTS[y]), x);
        // it is less OR EQUAL because if 0 is the only valid thread it must survive
__threadfence();
        if (maxTS[y] == x) {
            currentResult[y].infos[state] = stack[id + offset];
#pragma unroll
            for (int k=sequenceLen-1; k>state; k--) {
                currentResult[y].infos[k] = prevResults[y].infos[k];
            }
        }
    }
}


#ifdef CHECK_NEG_PARALLEL
__global__ void checkN(passer stackpage, EventInfoSet *prevResults, uint8_t *alive, int state, int minId, int negMinId, int size, GPUNegation *negations, int negNumber, passer negsPages, Parameter *param, parPasser paramSize, uint8_t aliveChecker) {
    int x = blockIdx.x*blockDim.x+threadIdx.x; //NEW EVT
    int y = blockIdx.y; //PARTIAL EVT
    int z = (blockIdx.z + negMinId) % ALLOC_SIZE; //NEGATION
    int id = (minId+x)%ALLOC_SIZE;
    int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;
    int negsOffset = negsPages.pageIdx[z/PAGE_SIZE] * PAGE_SIZE;
    int negsId = z % PAGE_SIZE;
    int aliveIdx = id + y*MAX_NEW_EVENTS;

    id %= PAGE_SIZE;

    if (x>=size) return;

    uint64_t negTimestamp = stack[negsId + negsOffset].timestamp;
    uint64_t maxTS, minTS;
    maxTS = prevResults[y].infos[negations[negNumber].upperId].timestamp;

    if (negations[negNumber].lowerId < 0)	{
        minTS = maxTS - negations[negNumber].lowerTime;
    } else {
        minTS = stack[id + offset].timestamp; // partialResult->indexes[neg->lowerId]->getTimeStamp();
    }
    //maxTS and minTS negation events are not valid; Jan 2015
    if (negTimestamp >= maxTS || negTimestamp <= minTS) return;

    int lres=1;
#pragma unroll
    for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
        int pp=paramSize.paramNum[p];
        if (pp==-1) break;
        lres *= d_checkComplexParameterForNegation(ev1, negsId, negsOffset, &param[pp], &prevResults, true, state);
    }

    if (lres!=0) alive[aliveIdx] = aliveChecker;
}
#endif

//this works only if it's run all in 1 single block!
__global__ void reduceFinal(EventInfoSet *prevResults, EventInfoSet *currentResult, int *currentIndex, int state, int sequenceLen) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x >= MAX_GEN || prevResults[x].infos[state].timestamp == 0) return;
    int writeIndex = atomicAdd(currentIndex, 1);
    if (writeIndex >= MAX_GEN) return;

#pragma unroll
    for (int k=sequenceLen; k>state; k--) {
        currentResult[writeIndex].infos[k] = prevResults[x].infos[k];
    }

}

__global__ void computeComplexEventsMultipleWithNegations(passer stackpage, EventInfoSet *prevResults, EventInfoSet *currentResult, int *currentIndex, uint8_t *alive, int state, int minId, int size, GPUParameter *param, parPasser paramSize, int sequenceLen, uint64_t win, int referredState, uint8_t aliveChecker, GPUNegation *negations, int negsNum, negsPasser negPages, GPUParameter *nParam) {
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y;
    int id = (minId+x)%ALLOC_SIZE;

     #ifdef CHECK_NEG_PARALLEL
    int aliveIdx = id + y*MAX_NEW_EVENTS;
#endif
    
    if (x>=size
#ifdef CHECK_NEG_PARALLEL
      || alive[aliveIdx]==aliveChecker
#endif
    )
      return; // Out of range
    int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;

    id %= PAGE_SIZE;

    uint64_t timestamp = stack[id + offset].timestamp;
    int valid = (timestamp < prevResults[y].infos[referredState].timestamp && timestamp > prevResults[y].infos[referredState].timestamp-win);

    if (paramSize.paramNum[0] != -1) {
#pragma unroll
        for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
            int pp=paramSize.paramNum[p];
            if (pp==-1) break;
            valid *= d_checkComplexParameter(id, offset, &param[pp], &prevResults[y], false, state);
        }
    }

    if (valid!=0 
      #ifndef CHECK_NEG_PARALLEL
      && gcheckN(&stack[id + offset], prevResults[y], negations, negsNum, negPages, nParam, state)
      #endif
    ) {
        int writeIndex = atomicAdd(currentIndex, 1);

        if (writeIndex >= MAX_GEN) return;

        currentResult[writeIndex].infos[state] = stack[offset + id];
#pragma unroll
        for (int k=sequenceLen-1; k>state; k--) {
            currentResult[writeIndex].infos[k] = prevResults[y].infos[k];
        }
    }
}

__global__ void computeComplexEventsMultiple(passer stackpage, EventInfoSet *prevResults, EventInfoSet *currentResult, int *currentIndex, int state, int minId, int size, GPUParameter *param, parPasser paramSize, int sequenceLen, uint64_t win, int referredState) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=size) return; // Out of range
    int j = blockIdx.y;
    int id = (minId+i)%ALLOC_SIZE;
    int offset = stackpage.pageIdx[id/PAGE_SIZE] * PAGE_SIZE;

    id %= PAGE_SIZE;

    uint64_t timestamp = stack[id + offset].timestamp;
    int valid = (timestamp < prevResults[j].infos[referredState].timestamp && timestamp > prevResults[j].infos[referredState].timestamp-win);

    if (paramSize.paramNum[0] != -1) {
#pragma unroll
        for (int p=0; p<MAX_PARAMETERS_NUM; p++) {
            int pp=paramSize.paramNum[p];
            if (pp==-1) break;
            valid *= d_checkComplexParameter(id, offset, &param[pp], &prevResults[j], false, state);
        }
    }

    if (valid!=0) {
        int writeIndex = atomicAdd(currentIndex, 1);

        if (writeIndex >= MAX_GEN) return;

        currentResult[writeIndex].infos[state] = stack[offset + id];
#pragma unroll
        for (int k=sequenceLen-1; k>state; k--) {
            currentResult[writeIndex].infos[k] = prevResults[j].infos[k];
        }
    }
}