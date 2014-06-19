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



/**
 * The constructor for MemoryManager is here because CUDA 5.0 wants __constant__ definitions to be accessed from one file only; this file is imported in CudaKernels.cu
 */
MemoryManager::MemoryManager()
{
    notified = false;
    totalMem = MAX_SIZE * 1048576;
    usedMem = 0;
    // resPtr + alivePtr + seqMaxPtr + prevResultPtr + currentResultPtr + resultsSizePtr + lastEventPtr
    ruleFixedSize = sizeof(int) + sizeof(int) + sizeof(GPUParameter)*3*MAX_PARAMETERS_NUM + sizeof(GPUNegation) * MAX_NEGATIONS_NUM;
    //eventsPtr
    workspaceFixedSize = sizeof(float)*MAX_GEN + sizeof(int)*MAX_GEN + 2*sizeof(EventInfoSet)*MAX_GEN + sizeof(uint8_t)*MAX_GEN*MAX_NEW_EVENTS;
    int e=0;
    for (int i=0; i<MAX_CONCURRENT_RULES; i++) {
        e += cudaMalloc((void**) &(workspaces[i].prevResultsPtr), (size_t) sizeof(EventInfoSet)*MAX_GEN);
        e += cudaMalloc((void**) &(workspaces[i].currentResultsPtr), (size_t) sizeof(EventInfoSet)*MAX_GEN);
        e += cudaMalloc((void**) &(workspaces[i].seqMaxPtr), (size_t) sizeof(int) * MAX_GEN);
        e += cudaMalloc((void**) &(workspaces[i].alivePtr), (size_t) sizeof(uint8_t) * MAX_GEN * MAX_NEW_EVENTS);
        e += cudaMalloc((void**) &(workspaces[i].aggregatesValPtr), (size_t) sizeof(float)*MAX_GEN);
    }
    usedMem += workspaceFixedSize * MAX_CONCURRENT_RULES;

    for (int i=0; i<MAX_RULE_NUM; i++) {
      
        swapped[i] = inUse[i] = false;
        rules[i].workspaceNum = i % MAX_CONCURRENT_RULES;
      
	//INITIALIZATION OF MEMORY STRUCTS
	for (int r=0; r<MAX_RULE_FIELDS-1; r++) {
            for (int z=0; z<MAX_PAGES_PER_STACK; z++) {
                rules[i].eventsChunks[r][z] = -1;
            }
        }
        for (int r=0; r<MAX_NEGATIONS_NUM; r++) {
            for (int z=0; z<MAX_PAGES_PER_STACK; z++) {
                rules[i].negationsChunks[r][z]=-1;
            }
        }
        for (int r=0; r<MAX_NUM_AGGREGATES; r++) {
            for (int z=0; z<MAX_PAGES_PER_STACK; z++) {
                rules[i].aggregatesChunks[r][z]=-1;
            }
        }
        //FIXED FOR EVERY RULE
        e += cudaMalloc((void**) &(rules[i].resPtr), (size_t) sizeof(int));
        e += cudaMalloc((void**) &(rules[i].resultsSizePtr), (size_t) sizeof(int));
        e += cudaMalloc((void**) &(rules[i].parametersPtr), (size_t) sizeof(GPUParameter)*MAX_PARAMETERS_NUM);
        e += cudaMalloc((void**) &(rules[i].aggregatesParametersPtr), (size_t) sizeof(GPUParameter)*MAX_PARAMETERS_NUM);
        e += cudaMalloc((void**) &(rules[i].negationsParametersPtr), (size_t) sizeof(GPUParameter)*MAX_PARAMETERS_NUM);
        e += cudaMalloc((void**) &(rules[i].negationsPtr), (size_t) sizeof(GPUNegation)*MAX_NEGATIONS_NUM);
    }
    usedMem += ruleFixedSize * MAX_RULE_NUM;
    if (usedMem > totalMem) {
        cout << "Not enough memory (" << usedMem/1024768 << " MB needed)" << endl;
        exit(-1);
    }

    uint64_t freeMem = totalMem - usedMem;
    chunksnum = freeMem / (sizeof(EventInfo) * PAGE_SIZE);
   
    d_chunksnum = chunksnum;
    chunksnum *= HOST_MEM_MUL;
    chunks = (chunk *)malloc(sizeof(chunk) * chunksnum);

    e += cudaMalloc((void**) &base_address, (size_t) sizeof(EventInfo) * PAGE_SIZE * chunksnum / HOST_MEM_MUL);
    e += cudaMallocHost((void**) &h_base_address, (size_t) sizeof(EventInfo) * PAGE_SIZE * chunksnum);


    for (int i=0; i<chunksnum; i++) {
        chunks[i].onGpu = -1;
        chunks[i].inUse = 0;
        chunks[i].pageIdx = i;
    }

    e += cudaMemcpyToSymbol(stack, &base_address, sizeof(EventInfo *));
    usedMem += sizeof(EventInfo) * PAGE_SIZE * chunksnum / HOST_MEM_MUL;
    if (e>0) {
        cout << "Error " << e << " allocating memory on the GPU!" << endl;
        cout << cudaGetErrorString((cudaError_t)e) << endl;
        exit(-1);
    }

//     cout << "GPU MemoryManager used " << usedMem / 1024768 << "MB" << endl;
}