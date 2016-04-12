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
#include <algorithm>

StacksRule::StacksRule(RulePkt* pkt)
    : eventGenerator(pkt->getCompositeEventTemplate()) {
  // Initializes the rule identifier
  ruleId = pkt->getRuleId();
  rulePkt = pkt;
  CompositeEventTemplate* templ = pkt->getCompositeEventTemplate();
  if (templ->getAttributesNum() + templ->getStaticAttributesNum() == 0) {
    compositeEventId = templ->getEventType();
  } else {
    compositeEventId = -1;
  }

  // Initialize stacks map with predicate and fills it with references
  int predicatesNum = pkt->getPredicatesNum();
  receivedPkts.resize(predicatesNum);
  referenceState.resize(predicatesNum, 0);
  for (int i = 0; i < predicatesNum; ++i) {
    Predicate& predicate = pkt->getPredicate(i);
    stacks.emplace_back(predicate.refersTo, predicate.win, predicate.kind);
    if (predicate.refersTo != -1) {
      stacks[predicate.refersTo].addLookBackTo(i);
      referenceState[i] = predicate.refersTo;
    }
  }

  // Initialize negations and fills stacks with references
  int negationsNum = pkt->getNegationsNum();
  receivedNegs.resize(negationsNum);
  for (int i = 0; i < negationsNum; ++i) {
    Negation& negation = pkt->getNegation(i);
    addNegation(negation.eventType, negation.constraints,
                negation.constraintsNum, negation.lowerId, negation.lowerTime,
                negation.upperId);
  }

  // Initialize aggregates belonging to the rule
  int aggregatesNum = pkt->getAggregatesNum();
  receivedAggs.resize(aggregatesNum);
  for (int i = 0; i < aggregatesNum; ++i) {
    Aggregate& aggregate = pkt->getAggregate(i);
    addAggregate(aggregate.eventType, aggregate.constraints,
                 aggregate.constraintsNum, aggregate.lowerId,
                 aggregate.lowerTime, aggregate.upperId, aggregate.name,
                 aggregate.fun);
  }

  // Initialize parameters belonging to the rule
  for (int i = 0; i < pkt->getParametersNum(); ++i) {
    Parameter& parameter = pkt->getParameter(i);
    addParameter(parameter.evIndex2, parameter.name2, parameter.evIndex1,
                 parameter.name1, parameter.type, pkt);
  }

  for (int i = 0; i < pkt->getComplexParametersNum(); ++i) {
    CPUParameter& parameter = pkt->getComplexParameter(i);
    addComplexParameter(parameter.operation, parameter.leftTree,
                        parameter.rightTree, parameter.lastIndex,
                        parameter.type, parameter.vtype);
  }
  // Initialize the set of consuming indexes
  consumingIndexes = pkt->getConsuming();
}

StacksRule::~StacksRule() {
  // Deletes stored messages used
  for (const auto& receivedPkt : receivedPkts) {
    for (const auto& pkt : receivedPkt) {
      if (pkt->decRefCount()) {
        delete pkt;
      }
    }
  }
  for (const auto& receivedAgg : receivedAggs) {
    for (const auto& pkt : receivedAgg) {
      if (pkt->decRefCount()) {
        delete pkt;
      }
    }
  }
  for (const auto& receivedNeg : receivedNegs) {
    for (const auto& pkt : receivedNeg) {
      if (pkt->decRefCount()) {
        delete pkt;
      }
    }
  }
}

void StacksRule::addToStack(PubPkt* pkt, int index) {
  parametricAddToStack(pkt, receivedPkts[index]);
}

void StacksRule::addToAggregateStack(PubPkt* pkt, int index) {
  parametricAddToStack(pkt, receivedAggs[index]);
}

void StacksRule::addToNegationStack(PubPkt* pkt, int index) {
  parametricAddToStack(pkt, receivedNegs[index]);
}

void StacksRule::startComputation(PubPkt* pkt, std::set<PubPkt*>& results) {
  // Adds the terminator to the last stack
  pkt->incRefCount();
  receivedPkts[0].push_back(pkt);
  // Removes expired events from stacks
  clearStacks();
  // Gets partial results (patterns)
  std::list<PartialEvent> partialResults = getPartialResults(pkt);
  // Checks parameters and removes invalid results from collected ones
  removePartialEventsNotMatchingParameters(partialResults, endStackParameters);
  // Creates complex events and adds them to the results
  createComplexEvents(partialResults, results);
  // Deletes consumed events
  removeConsumedEvent(partialResults);
  // Removes the terminator from the last stack
  receivedPkts[0].clear();
  if (pkt->decRefCount()) {
    delete pkt;
  }
}

void StacksRule::processPkt(PubPkt* pkt, MatchingHandler* mh,
                            std::set<PubPkt*>& results, int index) {
  std::map<int, std::set<int>>::iterator aggIt =
      mh->matchingAggregates.find(index);
  if (aggIt != mh->matchingAggregates.end()) {
    for (const auto& aggIndex : aggIt->second) {
      addToAggregateStack(pkt, aggIndex);
    }
  }
  std::map<int, std::set<int>>::iterator negIt =
      mh->matchingNegations.find(index);
  if (negIt != mh->matchingNegations.end()) {
    for (const auto& negIndex : negIt->second) {
      addToNegationStack(pkt, negIndex);
    }
  }
  std::map<int, std::set<int>>::iterator stateIt =
      mh->matchingStates.find(index);
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

void StacksRule::parametricAddToStack(PubPkt* pkt,
                                      std::vector<PubPkt*>& parReceived) {
  std::vector<PubPkt*>::iterator it =
      getBeginPacket(parReceived, pkt->getTimeStamp());
  parReceived.insert(it, pkt);
  pkt->incRefCount();
}

void StacksRule::addParameter(int index1, char* name1, int index2, char* name2,
                              StateType type, RulePkt* pkt) {
  Parameter tmp;
  tmp.evIndex1 = index1;
  tmp.evIndex2 = index2;
  tmp.type = type;
  strcpy(tmp.name1, name1);
  strcpy(tmp.name2, name2);
  if (type == STATE) {
    if (pkt->isInTheSameSequence(index1, index2) /*&& index2>0*/) {
      // FIXME what should go in here?
    } else {
      endStackParameters.push_back(std::move(tmp));
    }
  }
}

void StacksRule::addComplexParameter(Op pOperation, OpTree* lTree,
                                     OpTree* rTree, int lastIdx, StateType type,
                                     ValType vtype) {
  CPUParameter tmp;
  tmp.operation = pOperation;
  tmp.leftTree = lTree->dup();
  tmp.rightTree = rTree->dup();
  tmp.lastIndex = lastIdx;
  tmp.type = type;
  tmp.vtype = vtype;
  switch (type) {
    case STATE:
      branchStackComplexParameters[lastIdx].push_back(std::move(tmp));
      break;
    case NEG:
      negationComplexParameters[lastIdx].push_back(std::move(tmp));
      break;
    case AGG:
      aggregateComplexParameters[lastIdx].push_back(std::move(tmp));
      break;
  }
}

void StacksRule::addNegation(int eventType, Constraint constraints[],
                             int constrLen, int lowIndex, TimeMs& lowTime,
                             int highIndex) {
  Negation neg;
  neg.eventType = eventType;
  neg.constraintsNum = constrLen;
  neg.constraints = constraints;
  neg.lowerId = lowIndex;
  neg.lowerTime = lowTime;
  neg.upperId = highIndex;
  negations.push_back(std::move(neg));
  if (lowIndex < 0) {
    stacks[highIndex].addLinkedNegation(negations.size() - 1);
  } else {
    stacks[lowIndex].addLinkedNegation(negations.size() - 1);
  }
}

void StacksRule::addAggregate(int eventType, Constraint constraints[],
                              int constrLen, int lowIndex, TimeMs& lowTime,
                              int highIndex, char* parName, AggregateFun& fun) {
  Aggregate aggr;
  aggr.eventType = eventType;
  aggr.constraintsNum = constrLen;
  aggr.constraints = constraints;
  aggr.lowerId = lowIndex;
  aggr.lowerTime = lowTime;
  aggr.upperId = highIndex;
  aggr.fun = fun;
  strcpy(aggr.name, parName);
  aggregates.push_back(std::move(aggr));
}

void StacksRule::getWinEvents(std::list<PartialEvent>& results, int index,
                              TimeMs tsUp, CompKind mode,
                              PartialEvent& partialEvent) {
  bool useComplexParameters = false;

  std::vector<PubPkt*>& recPkts = receivedPkts[index];
  int stackSize = recPkts.size();
  Stack& stack = stacks[index];

  if (stackSize == 0) {
    return;
  }
  // Extracts the minimum and maximum element to process.
  // Returns immediately if they cannot be found.
  std::vector<PubPkt*>::iterator first =
      getBeginPacket(recPkts, tsUp - stack.getWindow());
  if (first == recPkts.end() || (*first)->getTimeStamp() >= tsUp) {
    return;
  }

  std::vector<PubPkt*>::iterator last = getEndPacket(recPkts, tsUp);
  if (first >= last) {
    return;
  }
  // Position the iterator to the last valid element
  --last;

  std::map<int, std::vector<CPUParameter>>::iterator itComplex =
      branchStackComplexParameters.find(index);
  if (itComplex != branchStackComplexParameters.end()) {
    useComplexParameters = true;
  }

  // In the case of a LAST_WITHIN semantics, reverses processing order
  if (mode == LAST_WITHIN) {
    std::swap(first, last);
  }

  while (true) {
    PubPkt* pkt = *first;
    if (!useComplexParameters ||
        checkParameters(pkt, partialEvent, itComplex->second, index, STATE)) {
      PartialEvent newPartialEvent;
      std::copy_n(partialEvent.indexes, stacks.size(), newPartialEvent.indexes);
      newPartialEvent.indexes[index] = pkt;
      // Check negations
      bool invalidatedByNegations = false;
      for (auto& neg : stack.getLinkedNegations()) {
        if (checkNegation(neg, newPartialEvent)) {
          invalidatedByNegations = true;
          break;
        }
      }
      // If it is not invalidated by events, add the new partial event to
      // results, otherwise delete it
      if (!invalidatedByNegations) {
        results.push_back(std::move(newPartialEvent));
        if (mode == LAST_WITHIN || mode == FIRST_WITHIN) {
          break;
        }
      }
    }
    // Updates index (increasing or decreasing, depending from the semantics)
    // and check termination condition
    if (mode == LAST_WITHIN) {
      --first;
      if (first < last) {
        break;
      }
    } else {
      ++first;
      if (first > last) {
        break;
      }
    }
  }
}

bool StacksRule::checkNegation(int negIndex, PartialEvent& partialResult) {
  Negation& neg = negations[negIndex];
  std::vector<PubPkt*>& recNeg = receivedNegs[negIndex];
  int negSize = recNeg.size();
  // No negations: return false
  if (negSize == 0) {
    return false;
  }
  // Extracts timestamps and indexes
  TimeMs maxTS = partialResult.indexes[neg.upperId]->getTimeStamp();
  TimeMs minTS = (neg.lowerId < 0)
                     ? maxTS - neg.lowerTime
                     : partialResult.indexes[neg.lowerId]->getTimeStamp();

  std::vector<PubPkt*>::iterator first = getBeginPacket(recNeg, minTS);

  // TODO: Aggiungere la seguente riga per avere uguaglianza semantica
  // con TRex nel test Rain.
  // if (recNeg[0]->getTimeStamp() <= maxTS &&
  //     recNeg[0]->getTimeStamp() >= minTS) {
  //   return true;
  // }

  // maxTS and minTS negation events are not valid; Jan 2015
  if (first == recNeg.end() || (*first)->getTimeStamp() >= maxTS) {
    return false;
  }

  std::vector<PubPkt*>::iterator last = getEndPacket(recNeg, maxTS);
  if (first >= last) {
    return false;
  }

  std::map<int, std::vector<CPUParameter>>::iterator itComp =
      negationComplexParameters.find(negIndex);
  if (itComp == negationComplexParameters.end()) {
    return true;
  }
  // If a negation can be found that satisfies all parameters,
  // then return true, otherwise return false
  return std::any_of(first, last, [&](PubPkt* pkt) {
    return checkParameters(pkt, partialResult, itComp->second, negIndex, NEG);
  });
}

std::list<PartialEvent> StacksRule::getPartialResults(PubPkt* pkt) {
  std::list<PartialEvent> prevEvents;
  std::list<PartialEvent> currentEvents;

  prevEvents.emplace_back();
  PartialEvent& last = prevEvents.back();
  last.indexes[0] = pkt;

  // Checks negations on the first state
  for (const auto& neg : stacks[0].getLinkedNegations()) {
    if (checkNegation(neg, last)) {
      return currentEvents;
    }
  }
  // Iterates over all states
  for (int state = 1; state < stacks.size(); ++state) {
    currentEvents.clear();
    // Iterates over all previously generated events
    for (auto& event : prevEvents) {
      // Extract events for next iteration
      int refState = referenceState[state];
      TimeMs maxTimeStamp = event.indexes[refState]->getTimeStamp();
      CompKind kind = stacks[state].getKind();
      getWinEvents(currentEvents, state, maxTimeStamp, kind, event);
    }
    // Swaps current and previous results
    std::swap(prevEvents, currentEvents);
    if (prevEvents.empty()) {
      break;
    }
  }
  return prevEvents;
}

bool StacksRule::checkParameter(PubPkt* pkt, PartialEvent& partialEvent,
                                Parameter& parameter) {
  PubPkt* receivedPkt = partialEvent.indexes[parameter.evIndex1];
  ValType type1, type2;
  int index1, index2;
  if (!receivedPkt->getAttributeIndexAndType(parameter.name2, index2, type2) ||
      !pkt->getAttributeIndexAndType(parameter.name1, index1, type1) ||
      type1 != type2) {
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

bool StacksRule::checkParameters(PubPkt* pkt, PartialEvent& partialEvent,
                                 std::vector<CPUParameter>& params, int index,
                                 StateType sType) {
  return std::all_of(params.begin(), params.end(), [&](CPUParameter& par) {
    return checkComplexParameter(pkt, &partialEvent, &par, index, sType);
  });
}

void StacksRule::removePartialEventsNotMatchingParameters(
    std::list<PartialEvent>& partialEvents,
    std::vector<Parameter>& parameters) {
  partialEvents.erase(
      std::remove_if(
          partialEvents.begin(), partialEvents.end(),
          [&](PartialEvent& pe) {
            return std::any_of(
                parameters.begin(), parameters.end(), [&](Parameter& par) {
                  return !checkParameter(pe.indexes[par.evIndex2], pe, par);
                });
          }),
      partialEvents.end());
}

void StacksRule::createComplexEvents(std::list<PartialEvent>& partialEvents,
                                     std::set<PubPkt*>& results) {
  for (auto& pe : partialEvents) {
    PubPkt* genPkt = NULL;
    if (compositeEventId >= 0) {
      genPkt = new PubPkt(compositeEventId, NULL, 0);
      genPkt->setTime(receivedPkts[0][0]->getTimeStamp());
    } else {
      genPkt = eventGenerator.generateCompositeEvent(
          pe, aggregates, receivedPkts, receivedAggs,
          aggregateComplexParameters);
    }
    results.insert(genPkt);
  }
}

void StacksRule::removeConsumedEvent(std::list<PartialEvent>& partialEvents) {
  if (consumingIndexes.empty()) {
    return;
  }

  for (int i : consumingIndexes) {
    std::set<PubPkt*> pktsToRemove;
    for (const auto& pe : partialEvents) {
      pktsToRemove.insert(pe.indexes[i]);
    }
    std::vector<PubPkt*>& recPkts = receivedPkts[i];
    for (auto it = recPkts.begin(); it != recPkts.end();) {
      PubPkt* pkt = *it;
      if (pktsToRemove.find(pkt) != pktsToRemove.end()) {
        it = recPkts.erase(it);
        if (pkt->decRefCount()) {
          delete pkt;
        }
      } else {
        ++it;
      }
    }
  }
}

void StacksRule::clearStacks() {
  for (int stack = 1; stack < stacks.size(); ++stack) {
    int refersToStack = stacks[stack].getRefersTo();
    if (receivedPkts[refersToStack].empty()) {
      continue;
    }
    TimeMs minTS = receivedPkts[refersToStack][0]->getTimeStamp() -
                   stacks[stack].getWindow();
    removeOldPacketsFromStack(minTS, receivedPkts[stack]);
  }
  for (int negIndex = 0; negIndex < negations.size(); ++negIndex) {
    Negation& neg = negations[negIndex];
    int refersToStack = neg.upperId;
    if (receivedPkts[refersToStack].empty()) {
      continue;
    }
    TimeMs win =
        neg.lowerId < 0 ? neg.lowerTime : stacks[neg.lowerId].getWindow();
    TimeMs minTS = receivedPkts[refersToStack][0]->getTimeStamp() - win;
    removeOldPacketsFromStack(minTS, receivedNegs[negIndex]);
  }
  for (int aggIndex = 0; aggIndex < aggregates.size(); ++aggIndex) {
    Aggregate& agg = aggregates[aggIndex];
    int refersToStack = agg.upperId;
    if (receivedPkts[refersToStack].empty()) {
      continue;
    }
    TimeMs win =
        agg.lowerId < 0 ? agg.lowerTime : stacks[agg.lowerId].getWindow();
    TimeMs minTS = receivedPkts[refersToStack][0]->getTimeStamp() - win;
    removeOldPacketsFromStack(minTS, receivedAggs[aggIndex]);
  }
}

void StacksRule::removeOldPacketsFromStack(TimeMs& minTS,
                                           std::vector<PubPkt*>& parReceived) {
  std::vector<PubPkt*>::iterator firstValid =
      getBeginPacket(parReceived, minTS);
  std::vector<PubPkt*>::iterator begin = parReceived.begin();
  for (auto it = begin; it != firstValid; ++it) {
    PubPkt* pkt = *it;
    if (pkt->decRefCount()) {
      delete pkt;
    }
  }
  parReceived.erase(begin, firstValid);
}
