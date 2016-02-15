//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Daniele Rogora
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

#include "GPUProcessor.h"

GPUProcessor::GPUProcessor(RulePkt* rulePkt, MemoryManager* m) {
  this->rulePkt = rulePkt;
  this->mm = m;
}

void GPUProcessor::getConstraint(int state, int num, int& idx, Op& o,
                                 int& val) {
  Constraint c =
      rulePkt->getPredicate(sequenceLen - 1 - state).constraints[num];
  o = c.op;
  val = c.intVal;
  idx = 0;
}

int GPUProcessor::getCompositeEventId() { return ceTemplate->getEventType(); }

CompositeEventTemplate* GPUProcessor::getCompositeEvent() { return ceTemplate; }

int GPUProcessor::getAttributesNum() { return ceTemplate->getAttributesNum(); }

int GPUProcessor::getStaticAttributesNum() {
  return ceTemplate->getStaticAttributesNum();
}

int GPUProcessor::getSequenceLen() { return sequenceLen; }

int GPUProcessor::getNumParam() { return rulePkt->getParametersNum(); }

CompKind GPUProcessor::getCompKind(int state) {
  return rulePkt->getPredicate(sequenceLen - 1 - state).kind;
}

uint64_t GPUProcessor::getWin(int state) {
  return rulePkt->getPredicate(sequenceLen - 1 - state).win.getTimeVal();
}

int GPUProcessor::getAggregatesNum() { return rulePkt->getAggregatesNum(); }

bool GPUProcessor::isConsuming(int state) {
  return consumingIndexes.find(state) != consumingIndexes.end();
}

int GPUProcessor::getNegationsNum() { return rulePkt->getNegationsNum(); }

Negation* GPUProcessor::getNegation(int num) { return negations[num]; }

int GPUProcessor::getNegsSize(int negIndex) { return negsSize[negIndex]; }

void GPUProcessor::addNegation(int eventType, Constraint constraints[],
                               int constrLen, int lowIndex, TimeMs& lowTime,
                               int highIndex) {
  negations[negsNum] = new Negation;
  negations[negsNum]->eventType = eventType;
  negations[negsNum]->constraintsNum = constrLen;
  negations[negsNum]->constraints = constraints;
  negations[negsNum]->lowerId = lowIndex;
  negations[negsNum]->lowerTime = lowTime;
  negations[negsNum]->upperId = highIndex;
  vector<PubPkt*> emptyVec;
  receivedNegs.insert(make_pair(negsNum, emptyVec));
  negsNum++;
}

void GPUProcessor::init() {
  aggregateParametersNum = 0;
  negationsParametersNum = 0;
  stateParamtersNum = 0;
  negsNum = 0;
  totevents = 0;
  first = true;
  this->sequenceLen = rulePkt->getPredicatesNum();

  ceTemplate = rulePkt->getCompositeEventTemplate()->dup();

  GPUProcessor* thisProc = this;
  mmToken = mm->subscribe(sequenceLen, rulePkt->getAggregatesNum(),
                          rulePkt->getNegationsNum(),
                          rulePkt->getComplexParametersNum());

  lowIndexes = (int*)malloc(sizeof(int) * sequenceLen);
  highIndexes = (int*)malloc(sizeof(int) * sequenceLen);
  if (rulePkt->getNegationsNum() > 0) {
    lowIndexesNegs = (int*)malloc(sizeof(int) * rulePkt->getNegationsNum());
    highIndexesNegs = (int*)malloc(sizeof(int) * rulePkt->getNegationsNum());
  }
  if (rulePkt->getAggregatesNum() > 0) {
    lowIndexesAggs = (int*)malloc(sizeof(int) * rulePkt->getAggregatesNum());
    highIndexesAggs = (int*)malloc(sizeof(int) * rulePkt->getAggregatesNum());
  }
  for (int i = 0; i < sequenceLen; i++) {

    lowIndexes[i] = 0;
    highIndexes[i] = 0;
    int topic = rulePkt->getPredicate(sequenceLen - 1 - i).eventType;
    map<int, set<int>>::iterator it = topics.find(topic);
    if (it == topics.end()) {
      set<int> emptySet;
      emptySet.insert(i);
      topics.insert(make_pair(topic, emptySet));
    } else {
      it->second.insert(i);
    }
  }

  // Initialize the set of consuming indexes
  set<int> cons = rulePkt->getConsuming();
  for (set<int>::iterator it = cons.begin(); it != cons.end(); ++it) {
    int consumedIndex = *it;
    consumingIndexes.insert(sequenceLen - consumedIndex - 1);
  }

  // Initialize negations and fills stacks with references
  for (int i = 0; i < rulePkt->getNegationsNum(); i++) {
    lowIndexesNegs[i] = 0;
    highIndexesNegs[i] = 0;
    negsSize[i] = 0;
    addNegation(
        rulePkt->getNegation(i).eventType, rulePkt->getNegation(i).constraints,
        rulePkt->getNegation(i).constraintsNum, rulePkt->getNegation(i).lowerId,
        rulePkt->getNegation(i).lowerTime, rulePkt->getNegation(i).upperId);

    map<int, set<int>>::iterator it =
        negationsTopics.find(rulePkt->getNegation(i).eventType);
    if (it == negationsTopics.end()) {
      set<int> emptySet;
      emptySet.insert(i);
      negationsTopics.insert(
          make_pair(rulePkt->getNegation(i).eventType, emptySet));
    } else {
      it->second.insert(i);
    }
  }

  aggsNum = 0;
  for (int i = 0; i < rulePkt->getAggregatesNum(); i++) {
    aggregates[i] = new Aggregate();
    aggregates[i]->constraints = rulePkt->getAggregate(i).constraints;
    aggregates[i]->constraintsNum = rulePkt->getAggregate(i).constraintsNum;
    aggregates[i]->eventType = rulePkt->getAggregate(i).eventType;
    aggregates[i]->lowerId = rulePkt->getAggregate(i).lowerId;
    aggregates[i]->lowerTime = rulePkt->getAggregate(i).lowerTime;
    aggregates[i]->upperId = rulePkt->getAggregate(i).upperId;
    strcpy(aggregates[i]->name, rulePkt->getAggregate(i).name);
    aggregates[i]->fun = rulePkt->getAggregate(i).fun;
    lowIndexesAggs[i] = 0;
    highIndexesAggs[i] = 0;
    int aggTopic = rulePkt->getAggregate(i).eventType;
    map<int, set<int>>::iterator it = aggregatesTopics.find(aggTopic);
    if (it == aggregatesTopics.end()) {
      set<int> emptySet;
      emptySet.insert(i);
      aggregatesTopics.insert(make_pair(aggTopic, emptySet));
    } else {
      it->second.insert(i);
    }
    aggsNum++;
  }
  int numParams = rulePkt->getComplexParametersNum();
  for (int i = 0; i < numParams; i++) {
    GPUParameter param = rulePkt->getComplexGPUParameter(i);
    addParameter(&param);
  }

  kernels = CudaKernels::getInstance(this, mm, mmToken);
  kernels->setProcessor(thisProc);
  if (!checkDefinitions())
    exit(-1);
}

bool GPUProcessor::checkDefinitions() {
  // Check that values in const.h are ok for this rule
  if (rulePkt->getAggregatesNum() > MAX_NUM_AGGREGATES) {
    cout << "You need to raise MAX_NUM_AGGREGATES to "
         << rulePkt->getAggregatesNum()
         << " to perform this computation. Exiting" << endl;
    return false;
  }
  if (rulePkt->getNegationsNum() > MAX_NEGATIONS_NUM) {
    cout << "You need to raise MAX_NEGATIONS_NUM to "
         << rulePkt->getNegationsNum()
         << " to perform this computation. Exiting" << endl;
    return false;
  }
  if (rulePkt->getPredicatesNum() > MAX_RULE_FIELDS) {
    cout << "You need to raise MAX_RULE_FIELDS to "
         << rulePkt->getPredicatesNum()
         << " to perform this computation. Exiting" << endl;
    return false;
  }
  if (negationsParametersNum > MAX_PARAMETERS_NUM ||
      aggregateParametersNum > MAX_PARAMETERS_NUM ||
      stateParamtersNum > MAX_PARAMETERS_NUM) {
    cout << "You need to raise MAX_PARAMETERS_NUM to perform this computation. "
            "Exiting" << endl;
    return false;
  }
  return true;
}

GPUProcessor::~GPUProcessor() {
  free(lowIndexes);
  free(highIndexes);
  if (rulePkt->getAggregatesNum() > 0) {
    free(lowIndexesAggs);
    free(highIndexesAggs);
  }
  if (rulePkt->getNegationsNum() > 0) {
    free(lowIndexesNegs);
    free(highIndexesNegs);
  }
  delete kernels;
}

void GPUProcessor::processEvent(PubPkt* event, set<PubPkt*>& generatedEvents) {
  list<int> states;
  list<int> aggs;
  list<int> negs;
  getMatchingColumns(event, topics, states);
  getMatchingColumns(event, aggregatesTopics, aggs);
  getMatchingColumns(event, negationsTopics, negs);

  for (list<int>::iterator it = states.begin(); it != states.end(); ++it) {
    int state = *it;
    if (!checkConstraints(event, rulePkt, sequenceLen - state - 1))
      return;
  }

  for (list<int>::iterator it = negs.begin(); it != negs.end(); ++it) {
    int neg = *it;
    if (states.size() == 0 && !checkNegationConstraints(event, rulePkt, neg))
      return;
    addEventToNegationStack(neg, event);
  }

  for (list<int>::iterator it = aggs.begin(); it != aggs.end(); ++it) {
    int agg = *it;
    if (states.size() == 0 && !checkAggregateConstraints(event, rulePkt, agg))
      return;
    addEventToAggregateStack(agg, event);
  }

  for (list<int>::iterator it = states.begin(); it != states.end(); ++it) {
    int state = *it;
    // The event matches an intermediate state
    if (state < sequenceLen - 1) {
      addEventToStack(state, event);
    }
    // The event matches the last state
    else {
      // Deletes expired events
      if (!deleteExpiredEvents(event))
        break;
      // If needed copies old infos to the kernel

      kernels->setEventsStacks();

      // Copies information on the GPU
      kernels->copyLastEventToDevice(event);

      if (first) {
        GPUProcessor* thisProc = this;
        samples[sequenceLen - 1] = event->copy();
        kernels->setProcessor(thisProc);
        kernels->setAggregates(aggregates);
        if (kernels->setParameters(parameters) &&
            kernels->setAggregatesParameters(aggregatesParameters) &&
            kernels->setNegationsParameters(negationsParameters))
          first = false;
      }
      // Starts the computation
      kernels->compute();
      // Gets the results
      kernels->getGeneratedEvents(generatedEvents);
    }
  }
#if MP_MODE == MP_COPY
  delete event;
#endif
}

uint64_t GPUProcessor::getMaxTimestamp(int stack, StateType t) {
  if (t == STATE) {
    // getFirstValidElementCircular(timestamps[stack],
    // lowIndexes[stack], highIndexes[stack], minTimestamp);
    return timestamps[stack][highIndexes[stack] - 1];
  } else if (t == AGG) {
    // return getFirstValidElementCircular(aggTimestamps[stack],
    // lowIndexesAggs[stack], highIndexesAggs[stack], minTimestamp);
    return aggTimestamps[stack][highIndexesAggs[stack] - 1];
  } else if (t == NEG) {
    return negTimestamps[stack][highIndexesNegs[stack] - 1];
  }
  // Suppresses gcc warning, should never be reached
  return -1;
}

int GPUProcessor::getLowIndex(int stack, StateType t) {
  if (t == STATE) {
    return lowIndexes[stack];
  } else if (t == AGG) {
    return lowIndexesAggs[stack];
  } else if (t == NEG) {
    return lowIndexesNegs[stack];
  }
  // Suppresses gcc warning, should never be reached
  return -1;
}

int GPUProcessor::getHighIndex(int stack, StateType t) {
  if (t == STATE) {
    return highIndexes[stack];
  } else if (t == AGG) {
    return highIndexesAggs[stack];
  } else if (t == NEG) {
    return highIndexesNegs[stack];
  }
  // Suppresses gcc warning, should never be reached
  return -1;
}

int GPUProcessor::getMinValidElement(int stack, uint64_t minTimestamp,
                                     StateType t) {
  if (t == STATE) {
    return getFirstValidElementCircular(timestamps[stack], lowIndexes[stack],
                                        highIndexes[stack], minTimestamp);
  } else if (t == AGG) {
    return getFirstValidElementCircular(aggTimestamps[stack],
                                        lowIndexesAggs[stack],
                                        highIndexesAggs[stack], minTimestamp);
  } else if (t == NEG) {
    return getFirstValidElementCircular(negTimestamps[stack],
                                        lowIndexesNegs[stack],
                                        highIndexesNegs[stack], minTimestamp);
  }
  // Suppresses gcc warning, should never be reached
  return -1;
}

// FIXME: controllare highIndexesAggs, a volte non conforme!
int GPUProcessor::getMaxValidElement(int stack, uint64_t maxTimestamp,
                                     StateType t) {
  if (t == STATE) {
    return getLastValidElementCircular(timestamps[stack], lowIndexes[stack],
                                       highIndexes[stack], maxTimestamp,
                                       lowIndexes[stack]);
  } else if (t == AGG) {
    return getLastValidElementCircular(
        aggTimestamps[stack], lowIndexesAggs[stack], highIndexesAggs[stack],
        maxTimestamp, lowIndexesAggs[stack]);
  } else if (t == NEG) {
    return getLastValidElementCircular(
        negTimestamps[stack], lowIndexesNegs[stack], highIndexesNegs[stack],
        maxTimestamp, lowIndexesNegs[stack]);
  }

  // Suppresses gcc warning, should never be reached
  return -1;
}

uint64_t GPUProcessor::getTimestamp(int stack, int index, StateType t) {
  if (t == STATE) {
    return timestamps[stack][index];
  } else if (t == AGG) {
    return aggTimestamps[stack][index];
  } else if (t == NEG) {
    return negTimestamps[stack][index];
  }
  // Suppresses gcc warning, should never be reached
  return -1;
}

int GPUProcessor::getNumParamAggregates() { return aggregateParametersNum; }

int GPUProcessor::getNumParamNegations() { return negationsParametersNum; }

void GPUProcessor::addParameter(GPUParameter* parameterp) {
  GPUParameter* parameter = (GPUParameter*)malloc(sizeof(GPUParameter));
  memcpy(parameter, parameterp, sizeof(GPUParameter));

  if (parameter->sType == STATE) {
    stateParamtersNum++;

    int state = parameter->lastIndex;

    map<int, set<GPUParameter*>>::iterator it = parameters.find(state);
    if (it == parameters.end()) {
      set<GPUParameter*> emptySet;
      emptySet.insert(parameter);
      parameters.insert(make_pair(state, emptySet));
    } else {
      it->second.insert(parameter);
    }
  }

  else if (parameter->sType == AGG) {
    aggregateParametersNum++;
    int state = parameter->aggIndex;
    map<int, set<GPUParameter*>>::iterator it =
        aggregatesParameters.find(state);
    if (it == aggregatesParameters.end()) {
      set<GPUParameter*> emptySet;
      emptySet.insert(parameter);
      aggregatesParameters.insert(make_pair(state, emptySet));
    } else {
      it->second.insert(parameter);
    }
  }

  else if (parameter->sType == NEG) {
    int state = parameter->negIndex;
    map<int, set<GPUParameter*>>::iterator it = negationsParameters.find(state);
    if (it == negationsParameters.end()) {
      set<GPUParameter*> emptySet;
      emptySet.insert(parameter);
      negationsParameters.insert(make_pair(state, emptySet));
    } else {
      it->second.insert(parameter);
    }
    negationsParametersNum++;
  }
}

int GPUProcessor::getRefersTo(int state) {
  return sequenceLen - 1 -
         rulePkt->getPredicate(sequenceLen - 1 - state).refersTo;
}

bool GPUProcessor::getAttributeIndexAndType(int state, char* name, int& index,
                                            ValType& type, StateType stype) {
  switch (stype) {
    case STATE:
      if (samples.find(sequenceLen - state - 1) == samples.end())
        return false;
      return samples[sequenceLen - state - 1]->getAttributeIndexAndType(
          name, index, type);
      break;

    case AGG:
      if (aggSamples.find(state) == aggSamples.end())
        return false;
      return aggSamples[state]->getAttributeIndexAndType(name, index, type);
      break;

    case NEG:
      if (negSamples.find(state) == negSamples.end())
        return false;
      return negSamples[state]->getAttributeIndexAndType(name, index, type);
      break;
  }

  return false;
}

bool GPUProcessor::compareEvt(EventInfo e1, EventInfo e2, int size) {
  if (e1.timestamp != e2.timestamp)
    return false;
  for (int i = 0; i < size; i++) {
    if (e1.attr[i].type != e2.attr[i].type)
      return false;
    switch (e1.attr[i].type) {
      case INT:
        if (e1.attr[i].intVal != e2.attr[i].intVal)
          return false;
        break;

      case FLOAT:
        if (e1.attr[i].floatVal != e2.attr[i].floatVal)
          return false;
        break;

      case BOOL:
        if (e1.attr[i].boolVal != e2.attr[i].boolVal)
          return false;
        break;

      case STRING:
        if (strcmp(e1.attr[i].stringVal, e2.attr[i].stringVal) != 0)
          return false;
        break;
    }
  }
  return true;
}

void GPUProcessor::remove(EventInfo ev, int stack) {
  for (int i = lowIndexes[stack]; i < highIndexes[stack];
       i = (i + 1) % ALLOC_SIZE) {
    if (compareEvt(mm->getEvent(mmToken, stack, i), ev,
                   samples[stack]->getAttributesNum())) {
      mm->cleanEventStack(mmToken, stack, i, highIndexes[stack]);
      break;
    }
  }
}

inline void GPUProcessor::addEventToStack(int stack, PubPkt* event) {
  int index = highIndexes[stack];

  if ((index + 1) % ALLOC_SIZE == lowIndexes[stack]) {
    // The stack is full!
    cout << "Stack full; enlarge ALLOC_SIZE" << endl;
    return;
  }

  if (index == 0) {
    samples[stack] = event->copy();
  }
  timestamps[stack][index] = event->getTimeStamp().getTimeVal();

  mm->addEventToStack(mmToken, createEventInfo(event), stack, index, STATE);

  highIndexes[stack] = (index + 1) % ALLOC_SIZE;
}

inline void GPUProcessor::addEventToAggregateStack(int stack, PubPkt* event) {
  int index = highIndexesAggs[stack];

  if ((index + 1) % ALLOC_SIZE == lowIndexesAggs[stack]) {
    // The stack is full!
    cout << "Aggregates stack full; enlarge ALLOC_SIZE" << endl;
    return;
  }

  if (index == 0) {
    aggSamples[stack] = event->copy();
  }
  aggTimestamps[stack][index] = event->getTimeStamp().getTimeVal();
  mm->addEventToStack(mmToken, createEventInfo(event), stack, index, AGG);
  highIndexesAggs[stack] = (index + 1) % ALLOC_SIZE;
}

inline void GPUProcessor::addEventToNegationStack(int stack, PubPkt* event) {
  int index = highIndexesNegs[stack];

  if ((index + 1) % ALLOC_SIZE == lowIndexesNegs[stack]) {
    // The stack is full!
    cout << "Negation Stack full; enlarge ALLOC_SIZE" << endl;
    return;
  }

  if (index == 0) {
    negSamples[stack] = event->copy();
  }
  negTimestamps[stack][index] = event->getTimeStamp().getTimeVal();
  mm->addEventToStack(mmToken, createEventInfo(event), stack, index, NEG);
  highIndexesNegs[stack] = (index + 1) % ALLOC_SIZE;
}

inline bool GPUProcessor::deleteExpiredEvents(PubPkt* event) {
  // States
  uint64_t minTimeStamp;
  for (int stack = sequenceLen - 2; stack >= 0; stack--) {
    int refersToStack = sequenceLen - 1 -
                        rulePkt->getPredicate(sequenceLen - stack - 1).refersTo;

    if (highIndexes[stack] - lowIndexes[stack] == 0)
      continue;
    if (refersToStack == sequenceLen - 1) {
      // It's the terminator!
      minTimeStamp =
          event->getTimeStamp().getTimeVal() -
          rulePkt->getPredicate(sequenceLen - 1 - stack).win.getTimeVal();
    } else {
      if (timestamps[refersToStack][lowIndexes[refersToStack]] >
          rulePkt->getPredicate(sequenceLen - 1 - stack).win.getTimeVal()) {
        minTimeStamp =
            timestamps[refersToStack][lowIndexes[refersToStack]] -
            rulePkt->getPredicate(sequenceLen - 1 - stack).win.getTimeVal();
      } else
        minTimeStamp = 0;
    }
    deleteInvalidElementsCircular(timestamps[stack], lowIndexes[stack],
                                  highIndexes[stack], minTimeStamp);
  }

  // Negations
  for (int negIndex = 0; negIndex < negsNum; negIndex++) {
    Negation* neg = negations[negIndex];
    int refersToStack = sequenceLen - neg->upperId - 1;
    uint64_t win;
    if (neg->lowerId < 0) {
      win = neg->lowerTime.getTimeVal();
    } else {
      int secondIndex = neg->lowerId;
      win = rulePkt->getPredicate(secondIndex).win.getTimeVal();
    }

    if (refersToStack == sequenceLen - 1) {
      // It's the terminator!
      minTimeStamp = event->getTimeStamp().getTimeVal() - win;
    } else {
      if (timestamps[refersToStack][lowIndexes[refersToStack]] > win) {
        minTimeStamp =
            timestamps[refersToStack][lowIndexes[refersToStack]] - win;
      } else
        minTimeStamp = 0;
    }
    deleteInvalidElementsCircular(negTimestamps[negIndex],
                                  lowIndexesNegs[negIndex],
                                  highIndexesNegs[negIndex], minTimeStamp);
  }

  // Aggregates
  for (int aggIndex = 0; aggIndex < aggsNum; aggIndex++) {
    Aggregate* agg = aggregates[aggIndex];
    int refersToStack = sequenceLen - agg->upperId - 1;
    uint64_t win = agg->lowerTime.getTimeVal();
    if (win < 0) {
      int secondIndex = agg->lowerId;
      win = rulePkt->getPredicate(secondIndex).win.getTimeVal();
    }

    if (refersToStack == sequenceLen - 1) {
      // It's the terminator!
      minTimeStamp = event->getTimeStamp().getTimeVal() - win;
    } else {
      if (timestamps[refersToStack][lowIndexes[refersToStack]] > win) {
        minTimeStamp =
            timestamps[refersToStack][lowIndexes[refersToStack]] - win;
      } else
        minTimeStamp = 0;
    }
    deleteInvalidElementsCircular(aggTimestamps[aggIndex],
                                  lowIndexesAggs[aggIndex],
                                  highIndexesAggs[aggIndex], minTimeStamp);
  }

  return true;
}
