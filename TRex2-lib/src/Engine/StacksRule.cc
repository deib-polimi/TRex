//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Alberto Negrello
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

#include "StacksRule.h"

using namespace std;

StacksRule::StacksRule(RulePkt* pkt) {
  // Initializes the rule identifier
  ruleId = pkt->getRuleId();
  rulePkt = pkt;
  eventGenerator =
      new CompositeEventGenerator(pkt->getCompositeEventTemplate());
  if (pkt->getCompositeEventTemplate()->getAttributesNum() +
          pkt->getCompositeEventTemplate()->getStaticAttributesNum() ==
      0) {
    compositeEventId = pkt->getCompositeEventTemplate()->getEventType();
  } else {
    compositeEventId = -1;
  }
  stacksNum = 0;
  aggrsNum = 0;
  negsNum = 0;
  // Initialize stacks map with predicate and fills it with references
  for (int i = 0; i < pkt->getPredicatesNum(); i++) {
    stacksSize[i] = 0;
    Stack* tmpStack =
        new Stack(pkt->getPredicate(i).refersTo, pkt->getPredicate(i).win,
                  pkt->getPredicate(i).kind);
    stacks.insert(make_pair(stacksNum, tmpStack));
    vector<PubPkt*> emptySet;
    receivedPkts.insert(make_pair(stacksNum, emptySet));
    stacksNum++;
    int refersTo = pkt->getPredicate(i).refersTo;
    if (refersTo != -1) {
      stacks[refersTo]->addLookBackTo(stacksNum - 1);
      referenceState.insert(make_pair(i, refersTo));
    }
  }
  // Initialize negations and fills stacks with references
  for (int i = 0; i < pkt->getNegationsNum(); i++) {
    negsSize[i] = 0;
    addNegation(pkt->getNegation(i).eventType, pkt->getNegation(i).constraints,
                pkt->getNegation(i).constraintsNum, pkt->getNegation(i).lowerId,
                pkt->getNegation(i).lowerTime, pkt->getNegation(i).upperId);
  }
  // Initialize aggregates belonging to the rule
  for (int i = 0; i < pkt->getAggregatesNum(); i++) {
    aggsSize[i] = 0;
    addAggregate(
        pkt->getAggregate(i).eventType, pkt->getAggregate(i).constraints,
        pkt->getAggregate(i).constraintsNum, pkt->getAggregate(i).lowerId,
        pkt->getAggregate(i).lowerTime, pkt->getAggregate(i).upperId,
        pkt->getAggregate(i).name, pkt->getAggregate(i).fun);
  }
  // Initialize parameters belonging to the rule
  for (int i = 0; i < pkt->getParametersNum(); i++) {
    addParameter(pkt->getParameter(i).evIndex2, pkt->getParameter(i).name2,
                 pkt->getParameter(i).evIndex1, pkt->getParameter(i).name1,
                 pkt->getParameter(i).type, pkt);
  }

  for (int i = 0; i < pkt->getComplexParametersNum(); i++) {
    addComplexParameter(pkt->getComplexParameter(i).operation,
                        pkt->getComplexParameter(i).leftTree,
                        pkt->getComplexParameter(i).rightTree,
                        pkt->getComplexParameter(i).lastIndex,
                        pkt->getComplexParameter(i).type,
                        pkt->getComplexParameter(i).vtype);
  }
  // Initialize the set of consuming indexes
  consumingIndexes = pkt->getConsuming();
}

StacksRule::~StacksRule() {
  // Deletes stored messages used
  for (const auto& receivedPkt : receivedPkts) {
    for (const auto& pkt : receivedPkt.second) {
      if (pkt->decRefCount()) {
        delete pkt;
      }
    }
  }
  for (const auto& receivedAgg : receivedAggs) {
    for (const auto& pkt : receivedAgg.second) {
      if (pkt->decRefCount()) {
        delete pkt;
      }
    }
  }
  for (const auto& receivedNeg : receivedNegs) {
    for (const auto& pkt : receivedNeg.second) {
      if (pkt->decRefCount()) {
        delete pkt;
      }
    }
  }

  // frees heap memory
  for (const auto& stack : stacks) {
    delete stack.second;
  }
  for (const auto& par : endStackParameters) {
    delete par;
  }

  for (const auto& branchStackComplexParameter : branchStackComplexParameters) {
    for (const auto& par : branchStackComplexParameter.second) {
      delete par;
    }
  }

  for (const auto& aggregateComplexParameter : aggregateComplexParameters) {
    for (const auto& par : aggregateComplexParameter.second) {
      delete par;
    }
  }
  for (const auto& aggregate : aggregates) {
    delete aggregate.second;
  }
  for (const auto& negation : negations) {
    delete negation.second;
  }
  delete eventGenerator;
}

void StacksRule::addToStack(PubPkt* pkt, int index) {
  parametricAddToStack(pkt, stacksSize[index], receivedPkts[index]);
}

void StacksRule::addToAggregateStack(PubPkt* pkt, int index) {
  parametricAddToStack(pkt, aggsSize[index], receivedAggs[index]);
}

void StacksRule::addToNegationStack(PubPkt* pkt, int index) {
  parametricAddToStack(pkt, negsSize[index], receivedNegs[index]);
}

void StacksRule::startComputation(PubPkt* pkt, set<PubPkt*>& results) {
  // Adds the terminator to the last stack
  pkt->incRefCount();
  receivedPkts[0].push_back(pkt);
  stacksSize[0] = 1;
  // Removes expired events from stacks
  clearStacks();
  // Gets partial results (patterns)
  list<PartialEvent*>* partialResults = getPartialResults(pkt);
  // Checks parameters and removes invalid results from collected ones
  removePartialEventsNotMatchingParameters(partialResults, endStackParameters);
  // Creates complex events and adds them to the results
  createComplexEvents(partialResults, results);
  // Deletes consumed events
  removeConsumedEvent(partialResults);
  // Deletes partial results
  deletePartialEvents(partialResults);
  // Removes the terminator from the last stack
  receivedPkts[0].clear();
  if (pkt->decRefCount()) {
    delete pkt;
  }
  stacksSize[0] = 0;
}

void StacksRule::processPkt(PubPkt* pkt, MatchingHandler* mh,
                            set<PubPkt*>& results, int index) {
  map<int, set<int>>::iterator aggIt = mh->matchingAggregates.find(index);
  if (aggIt != mh->matchingAggregates.end()) {
    for (const auto& aggIndex : aggIt->second) {
      addToAggregateStack(pkt, aggIndex);
    }
  }
  map<int, set<int>>::iterator negIt = mh->matchingNegations.find(index);
  if (negIt != mh->matchingNegations.end()) {
    for (const auto& negIndex : negIt->second) {
      addToNegationStack(pkt, negIndex);
    }
  }
  map<int, set<int>>::iterator stateIt = mh->matchingStates.find(index);
  if (stateIt != mh->matchingStates.end()) {
    bool lastStack = false;
    for (const auto& stateIndex : stateIt->second) {
      if (stateIndex != 0) {
        addToStack(pkt, stateIndex);
      } else {
        lastStack = true;
      }
    }
    if (lastStack) {
      startComputation(pkt, results);
    }
  }
}

void StacksRule::parametricAddToStack(PubPkt* pkt, int& parStacksSize,
                                      vector<PubPkt*>& parReceived) {
  TimeMs timeStamp = pkt->getTimeStamp();
  int i = getFirstValidElement(parReceived, parStacksSize, timeStamp);
  if (i == -1) {
    parStacksSize++;
    parReceived.push_back(pkt);
    pkt->incRefCount();
  } else {
    parStacksSize++;
    vector<PubPkt*>::iterator vecIt = parReceived.begin();
    parReceived.insert(vecIt + i, pkt);
    pkt->incRefCount();
  }
}

void StacksRule::addParameter(int index1, char* name1, int index2, char* name2,
                              StateType type, RulePkt* pkt) {
  Parameter* tmp = new Parameter;
  tmp->evIndex1 = index1;
  tmp->evIndex2 = index2;
  tmp->type = type;
  strcpy(tmp->name1, name1);
  strcpy(tmp->name2, name2);
  if (type == STATE) {
    if (pkt->isInTheSameSequence(index1, index2) /*&& index2>0*/) {
    } else {
      endStackParameters.insert(tmp);
    }
  }
}

void StacksRule::addComplexParameter(Op pOperation, OpTree* lTree,
                                     OpTree* rTree, int lastIdx, StateType type,
                                     ValType vtype) {
  CPUParameter* tmp = new CPUParameter;
  tmp->operation = pOperation;
  tmp->leftTree = lTree->dup();
  tmp->rightTree = rTree->dup();
  tmp->lastIndex = lastIdx;
  tmp->type = type;
  tmp->vtype = vtype;
  switch (type) {
    case STATE:
      branchStackComplexParameters[lastIdx].insert(tmp);
      break;
    case NEG:
      negationComplexParameters[lastIdx].insert(tmp);
      break;
    case AGG:
      aggregateComplexParameters[lastIdx].insert(tmp);
      break;
  }
}

void StacksRule::addNegation(int eventType, Constraint constraints[],
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
  if (lowIndex < 0) {
    stacks[highIndex]->addLinkedNegation(negsNum);
  } else {
    stacks[lowIndex]->addLinkedNegation(negsNum);
  }
  negsNum++;
}

void StacksRule::addAggregate(int eventType, Constraint* constraints,
                              int constrLen, int lowIndex, TimeMs& lowTime,
                              int highIndex, char* parName, AggregateFun& fun) {
  aggregates[aggrsNum] = new Aggregate;
  aggregates[aggrsNum]->eventType = eventType;
  aggregates[aggrsNum]->constraintsNum = constrLen;
  aggregates[aggrsNum]->constraints = constraints;
  aggregates[aggrsNum]->lowerId = lowIndex;
  aggregates[aggrsNum]->lowerTime = lowTime;
  aggregates[aggrsNum]->upperId = highIndex;
  aggregates[aggrsNum]->fun = fun;
  strcpy(aggregates[aggrsNum]->name, parName);
  vector<PubPkt*> emptyVec;
  receivedAggs.insert(make_pair(aggrsNum, emptyVec));
  aggrsNum++;
}

void StacksRule::getWinEvents(list<PartialEvent*>* results, int index,
                              TimeMs tsUp, CompKind mode,
                              PartialEvent* partialEvent) {
  bool useComplexParameters = false;
  if (stacksSize[index] == 0) {
    return;
  }
  // Extracts the minimum and maximum element to process.
  // Returns immediately if they cannot be found.
  TimeMs minTimeStamp = tsUp - stacks[index]->getWindow();
  int index1 = getFirstValidElement(receivedPkts[index], stacksSize[index],
                                    minTimeStamp);
  if (index1 < 0) {
    return;
  }
  if (receivedPkts[index][index1]->getTimeStamp() >= tsUp) {
    return;
  }
  int index2 =
      getLastValidElement(receivedPkts[index], stacksSize[index], tsUp, index1);
  if (index2 < 0) {
    index2 = index1;
  }
  map<int, set<CPUParameter*>>::iterator itComplex =
      branchStackComplexParameters.find(index);
  if (itComplex != branchStackComplexParameters.end()) {
    useComplexParameters = true;
  }

  // Computes the indexes for processing
  int count = 0;
  int endCount = index2 - index1;
  // In the case of a LAST_WITHIN semantics, reverses processing order
  if (mode == LAST_WITHIN) {
    count = index2 - index1;
    endCount = 0;
  }
  // Starts processing
  while (true) {
    bool usable = true;
    PubPkt* tmpPkt = receivedPkts[index][index1 + count];
    if (useComplexParameters) {
      usable = checkParameters(tmpPkt, partialEvent, itComplex->second, index,
                               STATE);
    }
    if (usable) {
      PartialEvent* newPartialEvent = new PartialEvent;
      memcpy(newPartialEvent->indexes, partialEvent->indexes,
             sizeof(PubPkt*) * stacksNum);
      newPartialEvent->indexes[index] = tmpPkt;
      // Check negations
      bool invalidatedByNegations = false;
      for (const auto& neg : *(stacks[index]->getLinkedNegations())) {
        if (checkNegation(neg, newPartialEvent)) {
          invalidatedByNegations = true;
          break;
        }
      }
      // If it is not invalidated by events, add the new partial event to
      // results, otherwise delete it
      if (!invalidatedByNegations) {
        results->push_back(newPartialEvent);
        if (mode == LAST_WITHIN || mode == FIRST_WITHIN) {
          break;
        }
      } else {
        delete newPartialEvent;
      }
    }
    // Updates index (increasing or decreasing, depending from the semantics)
    // and check termination condition
    if (mode == LAST_WITHIN) {
      count--;
      if (count < endCount) {
        return;
      }
    } else {
      count++;
      if (count > endCount) {
        return;
      }
    }
  }
}

bool StacksRule::checkNegation(int negIndex, PartialEvent* partialResult) {
  Negation* neg = negations[negIndex];
  // No negations: return false
  if (negsSize[negIndex] == 0) {
    return false;
  }
  // Extracts timestamps and indexes
  TimeMs maxTS = partialResult->indexes[neg->upperId]->getTimeStamp();
  TimeMs minTS;
  if (neg->lowerId < 0) {
    minTS = maxTS - neg->lowerTime;
  } else {
    minTS = partialResult->indexes[neg->lowerId]->getTimeStamp();
  }
  int index1 =
      getFirstValidElement(receivedNegs[negIndex], negsSize[negIndex], minTS);
  // TODO: Aggiungere la seguente riga per avere uguaglianza semantica
  // con TRex nel test Rain.
  // if (receivedNegs[negIndex][0]->getTimeStamp()<=maxTS &&
  // receivedNegs[negIndex][0]->getTimeStamp()>=minTS) return true;
  if (index1 < 0) {
    return false;
  }
  // maxTS and minTS negation events are not valid; Jan 2015
  if (receivedNegs[negIndex][index1]->getTimeStamp() >= maxTS) {
    return false;
  }
  int index2 = getLastValidElement(receivedNegs[negIndex], negsSize[negIndex],
                                   maxTS, index1);
  if (index2 < 0)
    index2 = index1;

  map<int, set<CPUParameter*>>::iterator itComplex =
      negationComplexParameters.find(negIndex);
  if (itComplex == negationComplexParameters.end()) {
    return true;
  }
  // Iterates over all negations and over all parameters.
  // If a negation can be found that satisfies all parameters,
  // then return true, otherwise return false
  for (int count = 0; count <= index2 - index1; count++) {
    PubPkt* tmpPkt = receivedNegs[negIndex].at(index1 + count);
    if (checkParameters(tmpPkt, partialResult, itComplex->second, negIndex,
                        NEG)) {
      return true;
    }
  }
  return false;
}

list<PartialEvent*>* StacksRule::getPartialResults(PubPkt* pkt) {
  list<PartialEvent*>* prevEvents = new list<PartialEvent*>;
  list<PartialEvent*>* currentEvents = new list<PartialEvent*>;
  PartialEvent* last = new PartialEvent;
  last->indexes[0] = pkt;
  prevEvents->push_back(last);
  // Checks negations on the first state
  for (const auto& neg : *(stacks[0]->getLinkedNegations())) {
    if (checkNegation(neg, last)) {
      delete last;
      delete prevEvents;
      return currentEvents;
    }
  }
  // Iterates over all states
  for (int state = 1; state < stacksNum; state++) {
    Stack* stack = stacks[state];
    // Iterates over all previously generated events
    for (const auto& event : *prevEvents) {
      // Extract events for next iteration
      int refState = referenceState[state];
      TimeMs maxTimeStamp = event->indexes[refState]->getTimeStamp();
      CompKind kind = stack->getKind();
      getWinEvents(currentEvents, state, maxTimeStamp, kind, event);
    }
    // Swaps current and previous results and deletes previous ones
    for (const auto& pe : *prevEvents) {
      delete pe;
    }
    prevEvents->clear();
    list<PartialEvent*>* temp = prevEvents;
    prevEvents = currentEvents;
    currentEvents = temp;
    if (prevEvents->empty()) {
      break;
    }
  }
  delete currentEvents;
  return prevEvents;
}

bool StacksRule::checkParameter(PubPkt* pkt, PartialEvent* partialEvent,
                                Parameter* parameter) {
  int indexOfReferenceEvent = parameter->evIndex1;
  PubPkt* receivedPkt = partialEvent->indexes[indexOfReferenceEvent];
  ValType type1, type2;
  int index1, index2;
  if (!receivedPkt->getAttributeIndexAndType(parameter->name2, index2, type2)) {
    return false;
  }
  if (!pkt->getAttributeIndexAndType(parameter->name1, index1, type1)) {
    return false;
  }
  if (type1 != type2) {
    return false;
  }
  switch (type1) {
    case INT:
      return receivedPkt->getIntAttributeVal(index2) ==
             pkt->getIntAttributeVal(index1);
    case FLOAT:
      return receivedPkt->getFloatAttributeVal(index2) ==
             pkt->getFloatAttributeVal(index1);
    case BOOL:
      return receivedPkt->getBoolAttributeVal(index2) ==
             pkt->getBoolAttributeVal(index1);
    case STRING:
      char result1[STRING_VAL_LEN];
      char result2[STRING_VAL_LEN];
      receivedPkt->getStringAttributeVal(index2, result2);
      pkt->getStringAttributeVal(index1, result1);
      return strcmp(result1, result2) == 0;
    default:
      return false;
  }
}

bool StacksRule::checkParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                 set<CPUParameter*>& complexParameters,
                                 int index, StateType sType) {
  for (const auto& par : complexParameters) {
    if (!checkComplexParameter(pkt, partialEvent, par, index, sType)) {
      return false;
    }
  }
  return true;
}

void StacksRule::removePartialEventsNotMatchingParameters(
    list<PartialEvent*>* partialEvents, set<Parameter*>& parameters) {
  for (auto it = partialEvents->begin(); it != partialEvents->end();) {
    PartialEvent* partialEvent = *it;
    bool valid = true;
    for (const auto& par : parameters) {
      int indexOfReferenceEvent = par->evIndex2;
      PubPkt* receivedPkt = partialEvent->indexes[indexOfReferenceEvent];
      if (!checkParameter(receivedPkt, partialEvent, par)) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      it = partialEvents->erase(it);
    } else {
      ++it;
    }
  }
}

void StacksRule::createComplexEvents(list<PartialEvent*>* partialEvents,
                                     set<PubPkt*>& results) {
  for (const auto& pe : *partialEvents) {
    PubPkt* genPkt = NULL;
    if (compositeEventId >= 0) {
      genPkt = new PubPkt(compositeEventId, NULL, 0);
      genPkt->setTime(receivedPkts[0][0]->getTimeStamp());
    } else {
      genPkt = eventGenerator->generateCompositeEvent(
          pe, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateComplexParameters);
    }
    results.insert(genPkt);
  }
}

void StacksRule::removeConsumedEvent(list<PartialEvent*>* partialEvents) {
  if (consumingIndexes.empty()) {
    return;
  }
  for (int i = 1; i < stacksNum; i++) {
    if (consumingIndexes.find(i) == consumingIndexes.end()) {
      continue;
    }
    set<PubPkt*> pktsToRemove;
    for (const auto& pe : *partialEvents) {
      PubPkt* pkt = pe->indexes[i];
      if (pktsToRemove.find(pkt) == pktsToRemove.end()) {
        pktsToRemove.insert(pkt);
      }
    }
    std::vector<PubPkt*>& recPkts = receivedPkts[i];
    for (auto it = recPkts.begin(); it != recPkts.end();) {
      PubPkt* pkt = *it;
      if (pktsToRemove.find(pkt) != pktsToRemove.end()) {
        it = recPkts.erase(it);
        if (pkt->decRefCount()) {
          delete pkt;
        }
        stacksSize[i]--;
      } else {
        ++it;
      }
    }
  }
}

void StacksRule::deletePartialEvents(list<PartialEvent*>* partialEvents) {
  for (const auto& pe : *partialEvents) {
    delete pe;
  }
  delete partialEvents;
}

void StacksRule::clearStacks() {
  for (int stack = 1; stack < stacksNum; stack++) {
    int refersToStack = stacks[stack]->getRefersTo();
    if (stacksSize[refersToStack] == 0) {
      continue;
    }
    TimeMs minTS = receivedPkts[refersToStack][0]->getTimeStamp() -
                   stacks[stack]->getWindow();
    removeOldPacketsFromStack(minTS, stacksSize[stack], receivedPkts[stack]);
  }
  for (int negIndex = 0; negIndex < negsNum; negIndex++) {
    Negation* neg = negations[negIndex];
    int refersToStack = neg->upperId;
    if (stacksSize[refersToStack] == 0) {
      continue;
    }
    TimeMs win;
    if (neg->lowerId < 0) {
      win = neg->lowerTime;
    } else {
      int secondIndex = neg->lowerId;
      win = stacks[secondIndex]->getWindow();
    }
    TimeMs minTS = receivedPkts[refersToStack][0]->getTimeStamp() - win;
    removeOldPacketsFromStack(minTS, negsSize[negIndex],
                              receivedNegs[negIndex]);
  }
  for (int aggIndex = 0; aggIndex < aggrsNum; aggIndex++) {
    Aggregate* agg = aggregates[aggIndex];
    int refersToStack = agg->upperId;
    if (stacksSize[refersToStack] == 0) {
      continue;
    }
    TimeMs win = agg->lowerTime;
    if (win < 0) {
      int secondIndex = agg->lowerId;
      win = stacks[secondIndex]->getWindow();
    }
    TimeMs minTS = receivedPkts[refersToStack][0]->getTimeStamp() - win;
    removeOldPacketsFromStack(minTS, aggsSize[aggIndex],
                              receivedAggs[aggIndex]);
  }
}

void StacksRule::removeOldPacketsFromStack(TimeMs& minTS, int& parStacksSize,
                                           vector<PubPkt*>& parReceived) {
  if (parStacksSize == 0) {
    return;
  }
  int newSize = deleteInvalidElements(parReceived, parStacksSize, minTS);
  if (newSize == parStacksSize) {
    return;
  }
  vector<PubPkt*>::iterator it = parReceived.begin();
  for (int count = 0; count < parStacksSize - newSize; count++) {
    PubPkt* pkt = *it;
    if (pkt->decRefCount()) {
      delete pkt;
    }
    it = parReceived.erase(it);
  }
  parStacksSize = newSize;
}
