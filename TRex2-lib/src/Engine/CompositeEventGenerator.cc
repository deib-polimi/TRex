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

#include "CompositeEventGenerator.h"

using namespace std;

CompositeEventGenerator::CompositeEventGenerator(
    CompositeEventTemplate* parCeTemplate) {
  ceTemplate = parCeTemplate->dup();
}

CompositeEventGenerator::~CompositeEventGenerator() { delete ceTemplate; }

PubPkt* CompositeEventGenerator::generateCompositeEvent(
    PartialEvent& partialEvent, std::vector<Aggregate>& aggregates,
    std::vector<int>& aggsSize, std::vector<std::vector<PubPkt*>>& receivedPkts,
    std::vector<std::vector<PubPkt*>>& receivedAggs,
    std::map<int, std::vector<CPUParameter>>& aggregateParameters) {
  int eventType = ceTemplate->getEventType();
  int attributesNum = ceTemplate->getAttributesNum();
  int staticAttributesNum = ceTemplate->getStaticAttributesNum();
  Attribute attributes[attributesNum + staticAttributesNum];
  for (int i = 0; i < attributesNum; i++) {
    ceTemplate->getAttributeName(attributes[i].name, i);
    ValType valType = ceTemplate->getAttributeTree(i)->getValType();
    attributes[i].type = valType;
    if (valType == INT)
      attributes[i].intVal = computeIntValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, ceTemplate->getAttributeTree(i));
    else if (valType == FLOAT)
      attributes[i].floatVal = computeFloatValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, ceTemplate->getAttributeTree(i));
    else if (valType == BOOL)
      attributes[i].boolVal = computeBoolValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, ceTemplate->getAttributeTree(i));
    else if (valType == STRING)
      computeStringValue(partialEvent, aggregates, aggsSize, receivedPkts,
                         receivedAggs, aggregateParameters,
                         ceTemplate->getAttributeTree(i),
                         attributes[i].stringVal);
  }
  for (int i = 0; i < staticAttributesNum; i++) {
    ceTemplate->getStaticAttribute(attributes[i + attributesNum], i);
  }
  PubPkt* result =
      new PubPkt(eventType, attributes, attributesNum + staticAttributesNum);
  result->setTime(partialEvent.indexes[0]->getTimeStamp());
  return result;
}

inline int CompositeEventGenerator::computeIntValue(
    PartialEvent& partialEvent, std::vector<Aggregate>& aggregates,
    std::vector<int>& aggsSize, std::vector<std::vector<PubPkt*>>& receivedPkts,
    std::vector<std::vector<PubPkt*>>& receivedAggs,
    std::map<int, std::vector<CPUParameter>>& aggregateParameters,
    OpTree* opTree) {
  OpTreeType type = opTree->getType();
  if (type == LEAF) {
    OpValueReference* reference = opTree->getValueReference();
    RulePktValueReference* pktReference =
        dynamic_cast<RulePktValueReference*>(reference);
    if (pktReference == NULL) {
      StaticValueReference* sReference =
          dynamic_cast<StaticValueReference*>(reference);
      if (sReference->getType() == INT)
        return sReference->getIntValue();
      else if (sReference->getType() == FLOAT)
        return sReference->getFloatValue();
      else if (sReference->getType() == BOOL)
        return sReference->getBoolValue();
    }
    int index = pktReference->getIndex();
    bool refersToAgg = pktReference->refersToAgg();
    if (!refersToAgg) {
      PubPkt* pkt = partialEvent.indexes[index];
      int attrIndex;
      ValType type;
      if (pkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                        type) == false)
        return 0;
      if (type == INT)
        return pkt->getIntAttributeVal(attrIndex);
      else if (type == FLOAT)
        return pkt->getFloatAttributeVal(attrIndex);
    } else {
      return computeAggregate(index, partialEvent, aggregates, aggsSize,
                              receivedPkts, receivedAggs, aggregateParameters);
    }
  } else {
    // Integer can only be obtained from integer:
    // assume this is ensured at rule deployment time
    int leftValue = computeIntValue(
        partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
        aggregateParameters, opTree->getLeftSubtree());
    int rightValue = computeIntValue(
        partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
        aggregateParameters, opTree->getRightSubtree());
    OpTreeOperation op = opTree->getOp();
    cout << "CPU not leaf " << leftValue << "; " << rightValue << ". OP " << op
         << endl;
    if (op == ADD)
      return leftValue + rightValue;
    if (op == SUB)
      return leftValue - rightValue;
    if (op == MUL)
      return leftValue * rightValue;
    if (op == DIV)
      return leftValue / rightValue;
  }
  return 0;
}

inline float CompositeEventGenerator::computeFloatValue(
    PartialEvent& partialEvent, std::vector<Aggregate>& aggregates,
    std::vector<int>& aggsSize, std::vector<std::vector<PubPkt*>>& receivedPkts,
    std::vector<std::vector<PubPkt*>>& receivedAggs,
    std::map<int, std::vector<CPUParameter>>& aggregateParameters,
    OpTree* opTree) {
  OpTreeType type = opTree->getType();
  if (type == LEAF) {
    OpValueReference* reference = opTree->getValueReference();
    RulePktValueReference* pktReference =
        dynamic_cast<RulePktValueReference*>(reference);
    if (pktReference == NULL) {
      // this is a static value
      StaticValueReference* sReference =
          dynamic_cast<StaticValueReference*>(reference);
      if (sReference->getType() == INT)
        return sReference->getIntValue();
      else if (sReference->getType() == FLOAT)
        return sReference->getFloatValue();
      else if (sReference->getType() == BOOL)
        return sReference->getBoolValue();
    }
    int index = pktReference->getIndex();
    bool refersToAgg = pktReference->refersToAgg();
    if (!refersToAgg) {
      PubPkt* pkt = partialEvent.indexes[index];
      int attrIndex;
      ValType type;
      if (pkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                        type) == false)
        return 0;
      if (type == INT)
        return pkt->getIntAttributeVal(attrIndex);
      else if (type == FLOAT)
        return pkt->getFloatAttributeVal(attrIndex);
    } else {
      return computeAggregate(index, partialEvent, aggregates, aggsSize,
                              receivedPkts, receivedAggs, aggregateParameters);
    }
  } else {
    // Floats can only be obtained from integer and float:
    // assume this is ensured at rule deployment time
    float leftValue;
    if (opTree->getLeftSubtree()->getValType() == INT)
      leftValue = computeIntValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, opTree->getLeftSubtree());
    else
      leftValue = computeFloatValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, opTree->getLeftSubtree());
    float rightValue;
    if (opTree->getRightSubtree()->getValType() == INT)
      rightValue = computeIntValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, opTree->getRightSubtree());
    else
      rightValue = computeFloatValue(
          partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
          aggregateParameters, opTree->getRightSubtree());
    OpTreeOperation op = opTree->getOp();
    if (op == ADD)
      return leftValue + rightValue;
    if (op == SUB)
      return leftValue - rightValue;
    if (op == MUL)
      return leftValue * rightValue;
    if (op == DIV)
      return leftValue / rightValue;
  }
  return 0;
}

inline bool CompositeEventGenerator::computeBoolValue(
    PartialEvent& partialEvent, std::vector<Aggregate>& aggregates,
    std::vector<int>& aggsSize, std::vector<std::vector<PubPkt*>>& receivedPkts,
    std::vector<std::vector<PubPkt*>>& receivedAggs,
    std::map<int, std::vector<CPUParameter>>& aggregateParameters,
    OpTree* opTree) {
  OpTreeType type = opTree->getType();
  if (type == LEAF) {
    OpValueReference* reference = opTree->getValueReference();
    RulePktValueReference* pktReference =
        dynamic_cast<RulePktValueReference*>(reference);
    if (pktReference == NULL) {
      // this is a static value
      StaticValueReference* sReference =
          dynamic_cast<StaticValueReference*>(reference);
      if (sReference->getType() == INT)
        return sReference->getIntValue();
      else if (sReference->getType() == FLOAT)
        return sReference->getFloatValue();
      else if (sReference->getType() == BOOL)
        return sReference->getBoolValue();
    }
    int index = pktReference->getIndex();
    bool refersToAgg = pktReference->refersToAgg();
    if (!refersToAgg) {
      PubPkt* pkt = partialEvent.indexes[index];
      int attrIndex;
      ValType type;
      if (pkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                        type) == false)
        return false;
      return pkt->getBoolAttributeVal(attrIndex);
    } else {
      // Aggregates not defines for type bool, up to now
      return false;
    }
  } else {
    // Booleans can only be obtained from booleans:
    // assume this is ensured at rule deployment time
    bool leftValue = computeBoolValue(
        partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
        aggregateParameters, opTree->getLeftSubtree());
    bool rightValue = computeBoolValue(
        partialEvent, aggregates, aggsSize, receivedPkts, receivedAggs,
        aggregateParameters, opTree->getRightSubtree());
    OpTreeOperation op = opTree->getOp();
    if (op == AND)
      return leftValue && rightValue;
    if (op == OR)
      return leftValue || rightValue;
  }
  return 0;
}

inline void CompositeEventGenerator::computeStringValue(
    PartialEvent& partialEvent, std::vector<Aggregate>& aggregates,
    std::vector<int>& aggsSize, std::vector<std::vector<PubPkt*>>& receivedPkts,
    std::vector<std::vector<PubPkt*>>& receivedAggs,
    std::map<int, std::vector<CPUParameter>>& aggregateParameters,
    OpTree* opTree, char* result) {
  // No operator is defined for strings: type can only be LEAF
  OpValueReference* reference = opTree->getValueReference();
  RulePktValueReference* pktReference =
      dynamic_cast<RulePktValueReference*>(reference);
  if (pktReference == NULL) {
    // this is a static value
    StaticValueReference* sReference =
        dynamic_cast<StaticValueReference*>(reference);
    return sReference->getStringValue(result);
  }
  int index = pktReference->getIndex();
  bool refersToAgg = pktReference->refersToAgg();
  if (!refersToAgg) {
    PubPkt* pkt = partialEvent.indexes[index];
    int attrIndex;
    ValType type;
    strcpy(result, "");
    if (pkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                      type) == false)
      return;
    pkt->getStringAttributeVal(attrIndex, result);
  } else {
    // Aggregates not defines for type string, up to now
  }
}

inline float CompositeEventGenerator::computeAggregate(
    int index, PartialEvent& partialEvent, std::vector<Aggregate>& aggregates,
    std::vector<int>& aggsSize, std::vector<std::vector<PubPkt*>>& receivedPkts,
    std::vector<std::vector<PubPkt*>>& receivedAggs,
    std::map<int, std::vector<CPUParameter>>& aggregateParameters) {
  Aggregate& agg = aggregates[index];
  TimeMs maxTS = partialEvent.indexes[agg.upperId]->getTimeStamp();
  TimeMs minTS = 0;
  if (agg.lowerId < 0) {
    minTS = maxTS - agg.lowerTime;
  } else {
    minTS = partialEvent.indexes[agg.lowerId]->getTimeStamp();
  }
  int index1 =
      getFirstValidElement(receivedAggs[index], aggsSize[index], minTS);
  if (index1 < 0)
    return 0;
  int index2 =
      getLastValidElement(receivedAggs[index], aggsSize[index], maxTS, index1);
  if (index2 < 0)
    index2 = index1;
  AggregateFun fun = agg.fun;
  char* name = agg.name;
  float sum = 0;
  int count = 0;
  float min = 0;
  float max = 0;
  int attIndex;
  ValType type;
  bool checkParams = false;
  bool firstValue = true;
  map<int, vector<CPUParameter>>::iterator paramIt =
      aggregateParameters.find(index);
  if (paramIt != aggregateParameters.end())
    checkParams = true;
  for (int i = index1; i <= index2; i++) {
    PubPkt* pkt = receivedAggs[index][i];
    if (checkParams) {
      if (!checkParameters(pkt, partialEvent, paramIt->second))
        continue;
      // if (! checkComplexParameter(pkt, partialEvent,
      // paramIt->second, -1, false)) continue;
    }
    float val = 0;
    if (pkt->getAttributeIndexAndType(name, attIndex, type) && type == INT) {
      val = pkt->getIntAttributeVal(attIndex);
    } else if (pkt->getAttributeIndexAndType(name, attIndex, type) &&
               type == FLOAT) {
      val = pkt->getFloatAttributeVal(attIndex);
    }
    count++;
    sum += val;
    // First value
    if (firstValue) {
      min = val;
      max = val;
      firstValue = false;
      continue;
    }
    // Following values
    if (val < min)
      min = val;
    else if (val > max)
      max = val;
  }
  if (fun == SUM)
    return sum;
  if (fun == MAX) {
    cout << "MAX: " << max << endl;
    return max;
  }
  if (fun == MIN)
    return min;
  if (fun == COUNT)
    return count;
  if (fun == AVG) {
    if (count == 0)
      return 0;
    else
      return sum / count;
  }
  return 0;
}

bool CompositeEventGenerator::checkParameters(
    PubPkt* pkt, PartialEvent& partialEvent, vector<CPUParameter>& parameters) {
  for (vector<CPUParameter>::iterator it = parameters.begin();
       it != parameters.end(); ++it) {
    // cout << "Agg par" << endl;
    if (!checkComplexParameter(pkt, &partialEvent, &(*it), -1, AGG))
      return false;
  }
  return true;
}
