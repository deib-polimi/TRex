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

// Power of 2, minimum is 1
#define MAX_BLOCKS 256
// Power of 2, minimum is 1
#define MIN_GPU_REDUCTION 8

void CudaKernels::getGeneratedEvents(set<PubPkt *> &generatedEventsSet) {
    if (*resultsSize>0) {
      setAggregateStacks();
        if (*resultsSize > MAX_GEN) {
            cout << "MAX_GEN too low; needed " << *resultsSize << " slots" << endl;
        }

        int e=cudaMemcpy(eventsSet, mm->getPrevResultsPtr(mmToken), (size_t) sizeof(EventInfoSet)*(*resultsSize), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        if (e>0) {
            cout << endl << " --- Error " << e << " while copying the event from the device!!!" << endl;
            exit(-1);
        }

        for (int i=0; i < *resultsSize; i++) {
	    staticAttributesAdded = false;
            PubPkt *genPkt = NULL;
            int compositeEventId = processor->getCompositeEventId();
            CompositeEventTemplate *ceTemplate = processor->getCompositeEvent();
            if (processor->getAttributesNum() + processor->getStaticAttributesNum() <= 0) {
                genPkt = new PubPkt(compositeEventId, NULL, 0);
                genPkt->setTime(eventsSet[i].infos[processor->getSequenceLen()-1].timestamp);
            } else {
                Attribute attributes[processor->getAttributesNum()+processor->getStaticAttributesNum()];
                for (int a=0; a<ceTemplate->getAttributesNum(); a++) {
                    ceTemplate->getAttributeName(attributes[a].name, a);
                    ValType valType = ceTemplate->getAttributeTree(a)->getValType();
                    attributes[a].type = valType;
		    if (valType==INT) attributes[a].intVal = computeIntValue(eventsSet[i], aggregates, ceTemplate->getAttributeTree(a), attributes);
		    else if (valType==FLOAT) attributes[a].floatVal = computeFloatValue(eventsSet[i], aggregates, ceTemplate->getAttributeTree(a), attributes);
		    else if (valType==BOOL) attributes[a].boolVal = computeBoolValue(eventsSet[i], aggregates, ceTemplate->getAttributeTree(a));
		    else if (valType==STRING) computeStringValue(eventsSet[i], aggregates, ceTemplate->getAttributeTree(a), attributes[a].stringVal);
                }
		if (processor->getStaticAttributesNum() > 0 && staticAttributesAdded == false) {
		  addStaticAttributes(ceTemplate, attributes);
		}
                genPkt = new PubPkt(compositeEventId, attributes, processor->getAttributesNum()+processor->getStaticAttributesNum());
                genPkt->setTime(eventsSet[i].infos[processor->getSequenceLen()-1].timestamp);

            }
            generatedEventsSet.insert(genPkt);  
        }
		deleteConsumed(eventsSet, *resultsSize);
    }
}


void CudaKernels::addStaticAttributes(CompositeEventTemplate *ceTemplate, Attribute *attributes)
{
	for (int i=0; i<ceTemplate->getStaticAttributesNum(); i++) {
		ceTemplate->getStaticAttribute(attributes[i+ceTemplate->getAttributesNum()], i);
	}
	staticAttributesAdded = true;
}


inline int CudaKernels::computeIntValue(EventInfoSet &events, map<int, Aggregate *> &aggregates, OpTree *opTree, Attribute *attributes) {
  	OpTreeType type = opTree->getType();
	if (type==LEAF) {
		OpValueReference *reference = opTree->getValueReference();
		RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
		if (pktReference == NULL) {
		  //this is a static value
		  StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
		  if (sReference->getType()==INT) return sReference->getIntValue();
		  else if (sReference->getType()==FLOAT) return sReference->getFloatValue();
		  else if (sReference->getType()==BOOL) return sReference->getBoolValue();
		}
		int index = pktReference->getIndex();
		if (! pktReference->refersToAgg()) {
		int attrIndex;
		ValType type;
		if (processor->getAttributeIndexAndType(pktReference->getIndex(), pktReference->getName(), attrIndex, type, STATE)==false) return 0;
		return events.infos[processor->getSequenceLen() - 1 - pktReference->getIndex()].attr[attrIndex].intVal;
		}
		return computeAggregates(events, aggregates, pktReference->getIndex(), opTree, attributes);
	} else {
		// Integer can only be obtained from integer: assume this is ensured at rule deployment time
		int leftValue = computeIntValue(events, aggregates, opTree->getLeftSubtree(), attributes);
		int rightValue = computeIntValue(events, aggregates, opTree->getRightSubtree(), attributes);
		OpTreeOperation op = opTree->getOp();
		if (op==ADD) return leftValue+rightValue;
		if (op==SUB) return leftValue-rightValue;
		if (op==MUL) return leftValue*rightValue;
		if (op==DIV) return leftValue/rightValue;
	}
	return 0;
}


inline float CudaKernels::computeFloatValue(EventInfoSet &events, map<int, Aggregate *> &aggregates, OpTree *opTree, Attribute *attributes) {
	OpTreeType type = opTree->getType();
	if (type==LEAF) {
		OpValueReference *reference = opTree->getValueReference();
		RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
		if (pktReference == NULL) {
		  //this is a static value
		  StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
		  if (sReference->getType()==INT) return sReference->getIntValue();
		  else if (sReference->getType()==FLOAT) return sReference->getFloatValue();
		  else if (sReference->getType()==BOOL) return sReference->getBoolValue();
		}
		int index = pktReference->getIndex();
		if (! pktReference->refersToAgg()) {
		int attrIndex;
		ValType type;
		if (processor->getAttributeIndexAndType(pktReference->getIndex(), pktReference->getName(), attrIndex, type, STATE)==false) return 0;
		return events.infos[processor->getSequenceLen() - 1 - pktReference->getIndex()].attr[attrIndex].floatVal;
		}
		return computeAggregates(events, aggregates, pktReference->getIndex(), opTree, attributes);
	} else {
		// Integer can only be obtained from integer: assume this is ensured at rule deployment time
		float leftValue;
		if (opTree->getLeftSubtree()->getValType()==INT) leftValue = computeIntValue(events, aggregates, opTree->getLeftSubtree(), attributes);
		else leftValue = computeFloatValue(events, aggregates, opTree->getLeftSubtree(), attributes);
		float rightValue;
		if (opTree->getRightSubtree()->getValType()==INT) rightValue = computeIntValue(events, aggregates, opTree->getRightSubtree(), attributes);
		else rightValue = computeFloatValue(events, aggregates, opTree->getRightSubtree(), attributes);
		OpTreeOperation op = opTree->getOp();
		if (op==ADD) return leftValue+rightValue;
		if (op==SUB) return leftValue-rightValue;
		if (op==MUL) return leftValue*rightValue;
		if (op==DIV) return leftValue/rightValue;
	}
	return 0;
}

 
inline bool CudaKernels::computeBoolValue(EventInfoSet &events, map<int, Aggregate *> &aggregates, OpTree *opTree) {
	OpTreeType type = opTree->getType();
	if (type==LEAF) {
		OpValueReference *reference = opTree->getValueReference();
		RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
		if (pktReference == NULL) {
		  //this is a static value
		  StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
		  if (sReference->getType()==INT) return sReference->getIntValue();
		  else if (sReference->getType()==FLOAT) return sReference->getFloatValue();
		  else if (sReference->getType()==BOOL) return sReference->getBoolValue();
		}
		int index = pktReference->getIndex();
		bool refersToAgg = pktReference->refersToAgg();
		if (! pktReference->refersToAgg()) {
		    int attrIndex;
		    ValType type;
		    if (processor->getAttributeIndexAndType(pktReference->getIndex(), pktReference->getName(), attrIndex, type, STATE)==false) return false;
		    return events.infos[processor->getSequenceLen() - 1 - pktReference->getIndex()].attr[attrIndex].boolVal;
		}
		else {
			// Aggregates not defined for type bool, up to now
			return false;
		}
	} else {
		// Booleans can only be obtained from booleans: assume this is ensured at rule deployment time
		bool leftValue = computeBoolValue(events, aggregates, opTree->getLeftSubtree());
		bool rightValue = computeBoolValue(events, aggregates, opTree->getRightSubtree());
		OpTreeOperation op = opTree->getOp();
		if (op==AND) return leftValue && rightValue;
		if (op==OR) return leftValue || rightValue;
	}
	return 0;
}
 
inline void CudaKernels::computeStringValue(EventInfoSet &events, map<int, Aggregate *> &aggregates, OpTree *opTree, char *result) {
 	// No operator is defined for strings: type can only be LEAF
 	OpValueReference *reference = opTree->getValueReference();
 	RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
	if (pktReference == NULL) {
		  //this is a static value
		  StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
		  return sReference->getStringValue(result);
		}
 	int index = pktReference->getIndex();
 	if (! pktReference->refersToAgg()) {
 		int attrIndex;
		ValType type;
		if (processor->getAttributeIndexAndType(pktReference->getIndex(), pktReference->getName(), attrIndex, type, STATE)==0) {
		  strcpy(result, "");
		  return;
		}
 		strcpy(result, events.infos[processor->getSequenceLen() - 1 - pktReference->getIndex()].attr[attrIndex].stringVal);
 	} else {
		// Aggregates not defined for type string, up to now
 	}
}

inline float CudaKernels::computeAggregates(EventInfoSet &usedEvents, map<int, Aggregate *> &aggregates, int i, OpTree *tree, Attribute *attributes) {
	cudaMemsetAsync(mm->getResultsSizePtr(mmToken), 0, (size_t) sizeof(int));
	CompositeEventTemplate *ceTemplate = processor->getCompositeEvent();
    
        OpValueReference *reference = tree->getValueReference();
        RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
        int index = pktReference->getIndex();
        bool refersToAgg = pktReference->refersToAgg();
	
	int aggId = i;
        Aggregate *agg = aggregates[i];
	int refState = agg->upperId; 
       
	
	int attrId;
	ValType type;
        if (processor->getAttributeIndexAndType(aggId, agg->name, attrId, type, AGG)==false) return 0;
        uint64_t maxTS = usedEvents.infos[processor->getSequenceLen()- refState - 1].timestamp;
        AggregateFun fun = agg->fun;

        uint64_t minTS = 0;
        if (agg->lowerId<0) {
            minTS = maxTS - agg->lowerTime.getTimeVal();
        } else {
            minTS = usedEvents.infos[processor->getSequenceLen()- agg->lowerId - 1].timestamp; 
        }
        
        int minId = processor->getMinValidElement(aggId, minTS, AGG);
        
        int maxId = processor->getMaxValidElement(aggId, maxTS, AGG);
        if (maxId<0) {
            maxId = minId;
        }
        int size;
        if (maxId>=minId) size=maxId-minId+1;
        else size=ALLOC_SIZE+maxId-minId+1;
	if (size==0 || minId<0) return 0;
        int n = 1;
        while (n<size) {
            n*=2;
        }
        int threads = 0;
        int blocks = 0;
        bool first = true;
        passer p;
        mm->getAggregatesIndexes(mmToken, aggId, p);

        // Perform reduction on the GPU until less than MIN_GPU_REDUCTION elements are left (and it is not the first round)
        while (n>MIN_GPU_REDUCTION || first) {
	    if (n==1) threads = 1;
            else if (n<GPU_THREADS*2) threads = n/2;
            else threads = GPU_THREADS;
            blocks = n/(threads*2);
            if (blocks == 0) blocks = 1;
            blocks = min(MAX_BLOCKS, blocks);
            parPasser parp;
            for (int b=0; b<MAX_PARAMETERS_NUM; b++) {
                parp.paramNum[b] = aggregatesParametersSize[aggId][b];
            }
            if (fun==SUM || fun==AVG || fun==COUNT) {
                if (first) {
                    first = false;
                    switch (threads) {
                    case 512: reduceFirstSumFloat<512><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case 256: reduceFirstSumFloat<256><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case 128: reduceFirstSumFloat<128><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case  64: reduceFirstSumFloat< 64><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case  32: reduceFirstSumFloat< 32><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case  16: reduceFirstSumFloat< 16><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case   8: reduceFirstSumFloat<  8><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case   4: reduceFirstSumFloat<  4><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case   2: reduceFirstSumFloat<  2><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    case   1: reduceFirstSumFloat<  1><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, mm->getResultsSizePtr(mmToken), fun); break;
                    }
                } else {
		  //No need for a second launch if we want only the COUNT; it's completed in the first launch!
		  if (fun==COUNT) break;
                    switch (threads) {
                    case 512: reduceSumFloat<512><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case 256: reduceSumFloat<256><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case 128: reduceSumFloat<128><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case  64: reduceSumFloat< 64><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case  32: reduceSumFloat< 32><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case  16: reduceSumFloat< 16><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case   8: reduceSumFloat<  8><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case   4: reduceSumFloat<  4><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case   2: reduceSumFloat<  2><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    case   1: reduceSumFloat<  1><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n); break;
                    }
                }
            }
            else if (fun==MAX || fun==MIN) {
                if (first) {
                    first = false;
                    switch (threads) {
                    case 512: reduceFirstMax<512><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case 256: reduceFirstMax<256><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case 128: reduceFirstMax<128><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case  64: reduceFirstMax< 64><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case  32: reduceFirstMax< 32><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case  16: reduceFirstMax< 16><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case   8: reduceFirstMax<  8><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case   4: reduceFirstMax<  4><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case   2: reduceFirstMax<  2><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    case   1: reduceFirstMax<  1><<<blocks, threads>>>(p, usedEvents, size, minId, refState, attrId, mm->getAggregatesValPtr(mmToken), mm->getAggregateParameterPtr(mmToken), parp, processor->getWin(refState), n, (fun==MAX), mm->getResultsSizePtr(mmToken)); break;
                    }
                } else {
                    switch (threads) {
                    case 512: reduceMax<512><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case 256: reduceMax<256><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case 128: reduceMax<128><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case  64: reduceMax< 64><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case  32: reduceMax< 32><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case  16: reduceMax< 16><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case   8: reduceMax<  8><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case   4: reduceMax<  4><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case   2: reduceMax<  2><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    case   1: reduceMax<  1><<<blocks, threads>>>(mm->getAggregatesValPtr(mmToken), mm->getAggregatesValPtr(mmToken), n, (fun==MAX)); break;
                    }
                }
            }
            n = blocks;
        }
        int e;
	e = cudaMemcpyAsync(aggRes, mm->getAggregatesValPtr(mmToken), (size_t) sizeof(float)*n, cudaMemcpyDeviceToHost);
        e = cudaMemcpyAsync(aggResultsSize, mm->getResultsSizePtr(mmToken), (size_t) sizeof(int), cudaMemcpyDeviceToHost);
        //Let's do this while the GPU is working!!
	addStaticAttributes(ceTemplate, attributes);
	//end of static attributes
        
	cudaDeviceSynchronize();
	// Performs the remaining part of the reduction on the CPU
        if (fun==MIN || fun==MAX) aggResult = aggRes[0];
        else aggResult = 0;
        
	
	cudaDeviceSynchronize();
        if (e>0) {
            cout << "Error " << e << " while reading aggregate value for aggregate " << aggId << "!!!" << endl;
            exit(-1);
        }

        for (int r=0; r<n; r++) {
            if (fun==SUM || fun==AVG) {
                aggResult += aggRes[r];
            }
            else if (fun==MAX) {
                if (aggRes[r] >  aggResult) aggResult = aggRes[r];
            }
            else if (fun==MIN) {
                if (aggRes[r] <  aggResult) aggResult = aggRes[r];
            }
        }
        
        if (fun==AVG) {
	  if (*aggResultsSize > 0) aggResult /= *aggResultsSize;
	  else aggResult = 0;
        }
        else if (fun==COUNT) {
                aggResult = *aggResultsSize;
        }
        return aggResult;
}