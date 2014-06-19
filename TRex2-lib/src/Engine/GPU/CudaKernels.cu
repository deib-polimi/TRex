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

#include "CudaKernels.h"
#include <unistd.h>
#include "Kernels/GPUFuns.cu"
#include "Kernels/Parameters.cu"
#include "Kernels/Aggregates.cu"
#include "Kernels/Computation.cu"
#include "MemoryManagerInit.cu"
#include "CompositeEventGenerator.cu"

CudaKernels *CudaKernels::instance = NULL;

CudaKernels::CudaKernels(GPUProcessorIf* processor, MemoryManager *m, int mmT) {
    refCount = 1;
    this->mm = m;
    mmToken = mmT;
    int e=0;
    
    e+=cudaMallocHost((void**) &eventsSet, (size_t) sizeof(EventInfoSet)*MAX_GEN);
    e+=cudaMallocHost((void**) &buffer, (size_t) sizeof(EventInfo));
    e+=cudaMallocHost((void**) &lastEvent, (size_t) sizeof(EventInfoSet));
    e+=cudaMallocHost((void**) &resultsSize, (size_t) sizeof(int));
    e+=cudaMallocHost((void**) &aggRes, (size_t) sizeof(float)*MIN_GPU_REDUCTION);
    e+=cudaMallocHost((void**) &aggResultsSize, (size_t) sizeof(int));
    
    if (e>0) {
        cout << endl << " --- Host Allocation Error!!! --- " << endl;
        exit(-1);
    }

}

CudaKernels::~CudaKernels() {
    mm->unsubscribe(mmToken, processor->getSequenceLen(), processor->getNegationsNum(), processor->getAggregatesNum());
    cudaFreeHost(eventsSet);
    cudaFreeHost(buffer);
    cudaFreeHost(lastEvent);
    cudaFreeHost(resultsSize);
    cudaFreeHost(aggRes);
}

CudaKernels * CudaKernels::getInstance(GPUProcessorIf* processor, MemoryManager *m, int mmT) {
    return new CudaKernels(processor,m, mmT);
}

void CudaKernels::decRefCount() {
    instance->refCount--;
    if (instance->refCount==0) {
        delete instance;
        instance = NULL;
    }
}

//Gets a reference to the calling GPUProcessor
void CudaKernels::setProcessor(GPUProcessorIf* processor) {
    this->processor = processor;
}

//Gets a reference to the aggregate's map of the calling GPUProcessor
void CudaKernels::setAggregates(map<int, Aggregate *> aggregates) {
    this->aggregates = aggregates;
}

bool CudaKernels::setParameters(map<int, set<GPUParameter *> > parameters) {
    for (int i=0; i < MAX_RULE_FIELDS - 1; i++) {
        for (int j=0; j<MAX_PARAMETERS_NUM; j++) {
            parametersSize[i][j] = -1;
        }
    }
    bool res;
    int j=0;

    for (int i=0; i<processor->getSequenceLen()-1; i++) {
        int k=0;
        map<int, set<GPUParameter *> >::iterator it=parameters.find(i);
        if (it!=parameters.end()) {
            int size = it->second.size();
            for (set<GPUParameter *>::iterator paramIt=it->second.begin(); paramIt!=it->second.end(); ++paramIt) {
                if (j >= MAX_PARAMETERS_NUM) {
                    cout << "Not enough parameter slots for states!" << endl;
                    exit(-1);
                }
                GPUParameter *param = *paramIt;
		for (int r=0; r < ( ( 1 << param->depth ) - 1); r++) {
		  if (!param->leftTree[r].empty && !param->leftTree[r].isStatic && param->leftTree[r].type==LEAF) {
		    res = processor->getAttributeIndexAndType(param->leftTree[r].refersTo, param->leftTree[r].attrName, param->leftTree[r].attrNum, param->leftTree[r].valueType, param->leftTree[r].sType);
		  }
		  else {
		    res = true;
		  }
		  if (!res) {
                    return false;
                }
		hostComplexParameters[j].leftTree[r] = param->leftTree[r];
		}
		for (int r=0; r < ( ( 1 << param->depth ) - 1); r++) {
		  if (!param->rightTree[r].empty && !param->rightTree[r].isStatic && param->rightTree[r].type==LEAF) {
		    res = processor->getAttributeIndexAndType(param->rightTree[r].refersTo, param->rightTree[r].attrName, param->rightTree[r].attrNum, param->rightTree[r].valueType, param->rightTree[r].sType);
		  }
		  else {
		    //devo popolare i nodi con i valori o ci sono già?
		    res = true;
		  }
		  if (!res) {
                    return false;
                }
		hostComplexParameters[j].rightTree[r] = param->rightTree[r];
		}
                parametersSize[i][k++] = j;
		hostComplexParameters[j].lSize = param->lSize;
		hostComplexParameters[j].rSize = param->rSize;
                hostComplexParameters[j].depth = param->depth;
		hostComplexParameters[j].lastIndex = param->lastIndex;
		hostComplexParameters[j].sType = param->sType;
		hostComplexParameters[j].vType = param->vType;
		hostComplexParameters[j].operation = param->operation;
		j++;
                
            }
        }
    }
    int e = cudaMemcpy(mm->getParameterPtr(mmToken), hostComplexParameters, (size_t) sizeof(GPUParameter)*MAX_PARAMETERS_NUM, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if (e>0) {
        cout << endl << " --- Error " << e << " while copying parameters in device memory!!! --- " << endl;
        exit(-1);
    }
    parametersNum = j;
    return true;
}

bool CudaKernels::setAggregatesParameters(map<int, set<GPUParameter *> > aggregatesParameters) {
    for (int i=0; i<MAX_RULE_FIELDS - 1; i++) {
        for (int j=0; j<MAX_PARAMETERS_NUM; j++) {
            aggregatesParametersSize[i][j] = -1;
        }
    }
    bool res;
    int j=0;
    GPUParameter hostComplexParameters[MAX_PARAMETERS_NUM];
    for (int i=0; i<processor->getSequenceLen()-1; i++) {
        int k=0;
        map<int, set<GPUParameter *> >::iterator it=aggregatesParameters.find(i);
        if (it!=aggregatesParameters.end()) {
            int size = it->second.size();
            for (set<GPUParameter *>::iterator paramIt=it->second.begin(); paramIt!=it->second.end(); ++paramIt) {
                if (j >= MAX_PARAMETERS_NUM) {
                    cout << "Not enough parameter slots for aggregates!" << endl;
                    exit(-1);
                }
                GPUParameter *param = *paramIt;
                for (int r=0; r < ( ( 1 << param->depth ) - 1); r++) {
		  if (!param->leftTree[r].empty && !param->leftTree[r].isStatic && param->leftTree[r].type==LEAF) {
		    res = processor->getAttributeIndexAndType(param->leftTree[r].refersTo, param->leftTree[r].attrName, param->leftTree[r].attrNum, param->leftTree[r].valueType, param->leftTree[r].sType);
		  }
		  else {
		    res = true;
		  }
		  if (!res) {
	            return false;
                }
		hostComplexParameters[j].leftTree[r] = param->leftTree[r];
		}
		for (int r=0; r < ( ( 1 << param->depth ) - 1); r++) {
		  if (!param->rightTree[r].empty && !param->rightTree[r].isStatic && param->rightTree[r].type==LEAF) {
		    res = processor->getAttributeIndexAndType(param->rightTree[r].refersTo, param->rightTree[r].attrName, param->rightTree[r].attrNum, param->rightTree[r].valueType, param->rightTree[r].sType);
		  }
		  else {
		    //devo popolare i nodi con i valori o ci sono già?
		    res = true;
		  }
		  if (!res) {
	            return false;
                }
		hostComplexParameters[j].rightTree[r] = param->rightTree[r];
		}
                aggregatesParametersSize[i][k++] = j;
		hostComplexParameters[j].lSize = param->lSize;
		hostComplexParameters[j].rSize = param->rSize;
                hostComplexParameters[j].depth = param->depth;
		hostComplexParameters[j].lastIndex = param->lastIndex;
		hostComplexParameters[j].sType = param->sType;
		hostComplexParameters[j].vType = param->vType;
		hostComplexParameters[j].operation = param->operation;
                j++;
            }
        }
    }
    int e = cudaMemcpy(mm->getAggregateParameterPtr(mmToken), hostComplexParameters, (size_t) sizeof(GPUParameter)*MAX_PARAMETERS_NUM, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if (e>0) {
        cout << endl << " --- Error " << e << " while copying parameters in device memory!!! --- " << endl;
        exit(-1);
    }
    return true;
}


bool CudaKernels::setNegationsParameters(map<int, set<GPUParameter *> > negationsParameters) {
    for (int n=0; n<processor->getNegationsNum(); n++) {

        Negation *neg = processor->getNegation(n);
        
        GPUNegation gn;
        if (neg->lowerId >= 0) gn.lowerId = processor->getSequenceLen()- neg->lowerId-1; //This is a negation between states
	else gn.lowerId = neg->lowerId; //this is a time based negation!
        gn.lowerTime = neg->lowerTime.getTimeVal();
        gn.upperId = processor->getSequenceLen()-neg->upperId-1;
        cudaMemcpy(mm->getNegationsPtr(mmToken) + n, &gn, sizeof(GPUNegation), cudaMemcpyHostToDevice);
    }


    for (int i=0; i<MAX_RULE_FIELDS - 1; i++) {
        for (int j=0; j<MAX_PARAMETERS_NUM; j++) {
            negationsParametersSize[i][j] = -1;
        }
    }

    bool res;
    int j=0;
    GPUParameter hostComplexParameters[MAX_PARAMETERS_NUM];
    for (int i=0; i<processor->getNumParamNegations(); i++) {
        int k=0;
        map<int, set<GPUParameter *> >::iterator it=negationsParameters.find(i);
        if (it!=negationsParameters.end()) {
            int size = it->second.size();
        
            for (set<GPUParameter *>::iterator paramIt=it->second.begin(); paramIt!=it->second.end(); ++paramIt) {
                if (j >= MAX_PARAMETERS_NUM) {
                    cout << "Not enough parameter slots for negations!" << endl;
                    exit(-1);
                }
                GPUParameter *param = *paramIt;
                for (int r=0; r < ( ( 1 << param->depth ) - 1); r++) {
		  if (!param->leftTree[r].empty && !param->leftTree[r].isStatic && param->leftTree[r].type==LEAF) {
		    res = processor->getAttributeIndexAndType(param->leftTree[r].refersTo, param->leftTree[r].attrName, param->leftTree[r].attrNum, param->leftTree[r].valueType, param->leftTree[r].sType);
		  }
		  else {
		    res = true;
		  }
		  if (!res) {
	            return false;
                }
		hostComplexParameters[j].leftTree[r] = param->leftTree[r];
		}
		for (int r=0; r < ( ( 1 << param->depth ) - 1); r++) {
		  if (!param->rightTree[r].empty && !param->rightTree[r].isStatic && param->rightTree[r].type==LEAF) {
		    res = processor->getAttributeIndexAndType(param->rightTree[r].refersTo, param->rightTree[r].attrName, param->rightTree[r].attrNum, param->rightTree[r].valueType, param->rightTree[r].sType);
		  }
		  else {
		    res = true;
		  }
		  if (!res) {
                    return false;
                }
		hostComplexParameters[j].rightTree[r] = param->rightTree[r];
		}
                negationsParametersSize[i][k++] = j;
		hostComplexParameters[j].lSize = param->lSize;
		hostComplexParameters[j].rSize = param->rSize;
                hostComplexParameters[j].depth = param->depth;
		hostComplexParameters[j].lastIndex = param->lastIndex;
		hostComplexParameters[j].sType = param->sType;
		hostComplexParameters[j].vType = param->vType;
		hostComplexParameters[j].operation = param->operation;
	        j++;        
            }
        }
    }

    int e = cudaMemcpy(mm->getNegationsParameterPtr(mmToken), hostComplexParameters, (size_t) sizeof(GPUParameter)*MAX_PARAMETERS_NUM, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if (e>0) {
        cout << endl << " --- Error " << e << " while copying parameters in device memory!!! --- " << endl;
        exit(-1);
    }
    return true;
}


void CudaKernels::setEventsStacks() {
    for (int i=0; i<processor->getSequenceLen()-1; i++) {
        int minId = processor->getLowIndex(i, STATE);
        int maxId = processor->getHighIndex(i, STATE);
        mm->setEventsStacks(mmToken, i, minId, maxId, STATE);
    }
    for (int i=0; i<processor->getNegationsNum(); i++) {
        int minId = processor->getLowIndex(i, NEG);
        int maxId = processor->getHighIndex(i, NEG);
        mm->setEventsStacks(mmToken, i, minId, maxId, NEG);
    }
}

void CudaKernels::setAggregateStacks() {
  for (int i=0; i<processor->getAggregatesNum(); i++) {
        int minId = processor->getLowIndex(i, AGG);
        int maxId = processor->getHighIndex(i, AGG);
        mm->setEventsStacks(mmToken, i , minId, maxId, AGG);
    }
}

void CudaKernels::copyLastEventToDevice(PubPkt *event) {
    mm->restorePointers(mmToken);
    lastEvent->infos[processor->getSequenceLen()-1] = createEventInfo(event);
    int e = cudaMemcpyAsync(mm->getPrevResultsPtr(mmToken), lastEvent, (size_t) sizeof(EventInfoSet), cudaMemcpyHostToDevice);
    if (e>0) {
        cout << endl << " --- Error " << e << " while copying the last event to the device!!! --- " << endl;
        exit(-1);
    }
    foots = event->getTimeStamp().getTimeVal();
}

LoopKind CudaKernels::prepareForNextLoop(int state, int &size, int &minId, passer &pages, parPasser &parameters, int negs[MAX_NEGATIONS_NUM], int &negsnum, bool &firstWithin) {
    LoopKind res;
    CompKind mode = processor->getCompKind(state);
    int maxId;
    minId = processor->getLowIndex(state, STATE);
    maxId = processor->getHighIndex(state, STATE);
    if (maxId>=minId) size = maxId-minId;
    else {
        size = ALLOC_SIZE+maxId-minId;
    }
    mm->getPageIndexes(mmToken, state, pages);
    int parnum=0;
    for (int b=0; b<MAX_PARAMETERS_NUM; b++) {
        parameters.paramNum[b] = parametersSize[state][b];
        if (parameters.paramNum[b] > -1) parnum++;
    }

    negsnum=0;
    // Checks negations on the first state
    if (state==processor->getSequenceLen()-2) {
        for (int n=0; n<processor->getNegationsNum(); n++) {
            Negation *neg = processor->getNegation(n);
            //Time based negation; consider upperId
            if (neg->lowerId < 0 && neg->upperId == 0) {
                negs[negsnum++] = n;
            }
        }
    }

    //And for the others
    for (int n=0; n<processor->getNegationsNum(); n++) {
        Negation *neg = processor->getNegation(n);
        //Negation between states; consider lowerId
        if (neg->lowerId >= 0 && processor->getSequenceLen() - neg->lowerId - 1 == state) {
            negs[negsnum++] = n;
        }
        //Time based negation; consider upperId
        else if (neg->lowerId < 0 && processor->getSequenceLen() - neg->upperId - 1 == state) {
            negs[negsnum++] = n;
        }
    }

    if (mode == LAST_WITHIN || mode == FIRST_WITHIN) {
        if (mode == LAST_WITHIN) firstWithin = false;
        else firstWithin = true;
        if (negsnum > 0) res = SINGLENEG;
        else {
            if (1+size/GPU_THREADS > 1) res = SINGLEGLOBAL;
            else res = SINGLE;
        }
    }

    else if (mode == EACH_WITHIN) {
        if (negsnum <= 0) res = MULTIPLE;
        else res = MULTIPLENEG;
    }
    return res;
}

void CudaKernels::setNegation(int n, passer &negp, parPasser &parp, int &negSize, int &negminId) {
    Negation *neg = processor->getNegation(n);
    mm->getNegationsIndexes(mmToken, n, negp);
    negminId = processor->getLowIndex(n, NEG);
    int negmaxId = processor->getHighIndex(n, NEG);
    if (negmaxId>=negminId) negSize = negmaxId-negminId;
    else negSize = ALLOC_SIZE+negmaxId-negminId;
    for (int b=0; b<MAX_PARAMETERS_NUM; b++) {
        parp.paramNum[b] = negationsParametersSize[n][b];
    }
}

void CudaKernels::setAllNegations(int i, int n, negsPasser &negp, int &negSize, int &negMinId) {
    mm->getAllNegationsIndexes(mmToken, n, i, negp);
    negMinId = processor->getLowIndex(n, NEG);
    int negmaxId = processor->getHighIndex(n, NEG);
    if (negmaxId>=negMinId) negSize = negmaxId-negMinId;
    else negSize = ALLOC_SIZE+negmaxId-negMinId;
//     cout << "Neg size " << negmaxId << "; " << negMinId << endl; 
    for (int b=0; b<MAX_PARAMETERS_NUM; b++) {
        negp.parameters[i][b] = negationsParametersSize[n][b];
    }
}

void CudaKernels::initAliveChecker() {
    aliveChecker = 1;
    #ifdef CHECK_NEG_PARALLEL
     cudaMemsetAsync(mm->getAlivePtr(mmToken), 0, (size_t) sizeof(uint8_t)*MAX_GEN*MAX_NEW_EVENTS);
#endif
}

void CudaKernels::compute() {
    int size, minId, colSize, negsNum;
    dim3 numBlocks, nBN;
    cudaError_t e;
    *resultsSize = 1;
    passer pages;
    parPasser parameters;
    int negations[MAX_NEGATIONS_NUM];
    negsPasser nP;

    LoopKind lkind;
    int state=processor->getSequenceLen()-2;
    bool firstWithin;
    lkind = prepareForNextLoop(state, size, minId, pages, parameters, negations, negsNum, firstWithin);

    for (; state>=0; state--) {
        cudaMemsetAsync(mm->getResultsSizePtr(mmToken), 0, (size_t) sizeof(int));
        if (aliveChecker==0) initAliveChecker();
        colSize = size/GPU_THREADS;
        if (size % GPU_THREADS > 0) colSize++;
        numBlocks = dim3(colSize, *resultsSize);
        if (size >= ALLOC_SIZE) {
            cout << "Not enough slots for alloc" << endl;
            exit(-1);
        }
	if (size==0) {
	  *resultsSize = 0;
	  return;
	}
	
	switch (lkind) {
        case MULTIPLE:
            computeComplexEventsMultiple<<<numBlocks, GPU_THREADS>>>(pages, mm->getPrevResultsPtr(mmToken), mm->getCurrentResultsPtr(mmToken), mm->getResultsSizePtr(mmToken), state, minId, size, mm->getParameterPtr(mmToken), parameters, processor->getSequenceLen(), processor->getWin(state), processor->getRefersTo(state));
            break;

        case SINGLE:
            computeComplexEventsSingle<<<numBlocks, GPU_THREADS>>>(pages, mm->getPrevResultsPtr(mmToken), mm->getCurrentResultsPtr(mmToken), mm->getResultsSizePtr(mmToken), state, minId, size, mm->getParameterPtr(mmToken), parameters, processor->getSequenceLen(), processor->getWin(state), processor->getRefersTo(state), firstWithin);
            break;


        case MULTIPLENEG:
            do {
                int partialSize = min(size, MAX_NEW_EVENTS);
                colSize = partialSize/GPU_THREADS;
                if (partialSize % GPU_THREADS > 0) colSize++;
                for (int n=0; n<negsNum; n++) {
#ifdef CHECK_NEG_PARALLEL
		     int negSize, negMinId;
                     setNegation(negations[n], negPages, negParameters, negSize, negMinId);
                     if (negSize > 0) {
                         nBN = dim3(colSize, *resultsSize, negSize);
                          checkN<<<nBN, GPU_THREADS>>>(pages, mm->getPrevResultsPtr(mmToken), mm->getAlivePtr(mmToken), state, minId, negMinId, partialSize, mm->getNegationsPtr(mmToken), negations[n], negPages, mm->getNegationsParameterPtr(mmToken), negParameters, aliveChecker);
                     }
#else
		  setAllNegations(n, negations[n], nP, nP.size[n], nP.minId[n]);
#endif
                }
                numBlocks = dim3(colSize, *resultsSize);
                computeComplexEventsMultipleWithNegations<<<numBlocks, GPU_THREADS>>>(pages, mm->getPrevResultsPtr(mmToken), mm->getCurrentResultsPtr(mmToken), mm->getResultsSizePtr(mmToken), mm->getAlivePtr(mmToken), state, minId, partialSize, mm->getParameterPtr(mmToken), parameters, processor->getSequenceLen(), processor->getWin(state), processor->getRefersTo(state), aliveChecker, mm->getNegationsPtr(mmToken), negsNum, nP, mm->getNegationsParameterPtr(mmToken));
                minId += partialSize;
                size -= partialSize;
            } while (size > 0);
            aliveChecker++;
            break;

        case SINGLEGLOBAL:
        case SINGLENEG:
            cudaMemsetAsync(mm->getCurrentResultsPtr(mmToken), 0, (size_t) sizeof(EventInfoSet)*MAX_GEN);
            do {
                int partialSize = min(size, MAX_NEW_EVENTS);
                colSize = partialSize/GPU_THREADS;
                if (partialSize % GPU_THREADS > 0) colSize++;
                if (!firstWithin)
                {
                    cudaMemsetAsync(mm->getSeqMaxPtr(mmToken), 0, (size_t) sizeof(int)*MAX_GEN);
                }
                else {
                    cudaMemsetAsync(mm->getSeqMaxPtr(mmToken), 127, (size_t) sizeof(int)*MAX_GEN);
                }
                for (int n=0; n<negsNum; n++) {
#ifdef CHECK_NEG_PARALLEL
		  int negSize, negMinId;
                     setNegation(negations[n], negPages, negParameters, negSize, negMinId);
                     if (negSize > 0) {
                         nBN = dim3(colSize, *resultsSize, negSize);
                          checkN<<<nBN, GPU_THREADS>>>(pages, mm->getPrevResultsPtr(mmToken), mm->getAlivePtr(mmToken), state, minId, negMinId, partialSize, mm->getNegationsPtr(mmToken), negations[n], negPages, mm->getNegationsParameterPtr(mmToken), negParameters, aliveChecker);
                     }
#else
		  setAllNegations(n, negations[n], nP, nP.size[n], nP.minId[n]);
#endif
                }
                numBlocks = dim3(colSize, *resultsSize);
                computeComplexEventsSingleG<<<numBlocks, GPU_THREADS>>>(pages, mm->getPrevResultsPtr(mmToken), mm->getCurrentResultsPtr(mmToken), mm->getResultsSizePtr(mmToken), mm->getSeqMaxPtr(mmToken), mm->getAlivePtr(mmToken), state, minId, partialSize, mm->getParameterPtr(mmToken), parameters, processor->getSequenceLen(), processor->getWin(state), processor->getRefersTo(state), firstWithin, aliveChecker, mm->getNegationsPtr(mmToken), negsNum, nP, mm->getNegationsParameterPtr(mmToken));
                minId += partialSize;
                size -= partialSize;
            } while (size > 0);
	    mm->swapPointers(mmToken);
            int bl = MAX_GEN/GPU_THREADS;
            if (MAX_GEN % GPU_THREADS > 0) bl++;
            reduceFinal<<<bl, GPU_THREADS>>>(mm->getPrevResultsPtr(mmToken), mm->getCurrentResultsPtr(mmToken), mm->getResultsSizePtr(mmToken), state, processor->getSequenceLen());
            mm->swapPointers(mmToken);
            aliveChecker++;
            break;

        }

        if (state > 0)	lkind = prepareForNextLoop(state-1, size, minId, pages, parameters, negations, negsNum, firstWithin);
        
	e=cudaMemcpyAsync(resultsSize, mm->getResultsSizePtr(mmToken), (size_t) sizeof(int), cudaMemcpyDeviceToHost);
        if (e>0) {
            cout << endl << " --- Error " << e << " while reading the number of results produced from device memory!!!" << endl;
            exit(-1);
        }

        cudaDeviceSynchronize();
        if (*resultsSize > MAX_GEN) {
            cout << "MAX_GEN too low; needed " << *resultsSize << " slots" << endl;
            exit(-1);
	}
        if (*resultsSize==0) {
            return;
        }
        else {
            //check next state
            mm->swapPointers(mmToken);
        }
    }

}

void CudaKernels::deleteConsumed(EventInfoSet *eventsSet, int size) {
    for (int j=0; j<processor->getSequenceLen(); j++) {
        if (processor->isConsuming(j)) {
            for (int i=0; i<size; i++) {
                processor->remove(eventsSet[i].infos[j], j);
            }
        }
    }
}


