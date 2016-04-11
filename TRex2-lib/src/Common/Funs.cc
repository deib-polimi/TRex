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

#include "Funs.h"
#include <algorithm>

using namespace std;

// This C function is needed to make the library compatible with the
// AC_CHECK_LIB macro
extern "C" {
void libTRex2_is_present(void){};
}

bool checkConstraints(PubPkt* event, RulePkt* rule, int state) {
  int idx, tempint;
  float tempfloat;
  bool tempbool;
  char tempString[STRING_VAL_LEN];
  ValType t;
  Op o;
  for (int k = 0; k < rule->getPredicate(state).constraintsNum; k++) {

    if (event->getAttributeIndexAndType(
            rule->getPredicate(state).constraints[k].name, idx, t) == false)
      return false;
    o = rule->getPredicate(state).constraints[k].op;
    switch (t) {
      case INT:
        tempint = rule->getPredicate(state).constraints[k].intVal;
        if (o == EQ) {
          if (event->getAttribute(idx).intVal != tempint)
            return false;
        } else if (o == GT) {
          if (event->getAttribute(idx).intVal <= tempint)
            return false;
        } else if (o == LT) {
          if (event->getAttribute(idx).intVal >= tempint)
            return false;
        } else if (o == LE) {
          if (event->getAttribute(idx).intVal > tempint)
            return false;
        } else if (o == GE) {
          if (event->getAttribute(idx).intVal < tempint)
            return false;
        }
        break;

      case FLOAT:
        tempfloat = rule->getPredicate(state).constraints[k].floatVal;
        if (o == EQ) {
          if (event->getAttribute(idx).floatVal != tempfloat)
            return false;
        } else if (o == GT) {
          if (event->getAttribute(idx).floatVal <= tempfloat)
            return false;
        } else if (o == LT) {
          if (event->getAttribute(idx).floatVal >= tempfloat)
            return false;
        } else if (o == LE) {
          if (event->getAttribute(idx).floatVal > tempfloat)
            return false;
        } else if (o == GE) {
          if (event->getAttribute(idx).floatVal < tempfloat)
            return false;
        }

        break;

      case BOOL:
        tempbool = rule->getPredicate(state).constraints[k].boolVal;
        if (o == EQ) {
          if (event->getAttribute(idx).boolVal != tempbool)
            return false;
        }
        if (o == NE) {
          if (event->getAttribute(idx).boolVal == tempbool)
            return false;
        } else {
          // Not defined
          return false;
        }
        break;

      case STRING:
        strcpy(tempString, rule->getPredicate(state).constraints[k].stringVal);
        if (o == EQ) {
          if (strcmp(event->getAttribute(idx).stringVal, tempString) != 0)
            return false;
        } else if (o == NE) {
          if (strcmp(event->getAttribute(idx).stringVal, tempString) == 0)
            return false;
        } else {
          // Not defined
          return false;
        }
        break;
    }
  }
  return true;
}

bool checkAggregateConstraints(PubPkt* event, RulePkt* rule, int aggNum) {
  int idx, tempint;
  float tempfloat;
  bool tempbool;
  char tempString[STRING_VAL_LEN];
  ValType t;
  Op o;
  for (int k = 0; k < rule->getAggregate(aggNum).constraintsNum; k++) {
    if (event->getAttributeIndexAndType(
            rule->getAggregate(aggNum).constraints[k].name, idx, t) == false)
      return false;
    o = rule->getAggregate(aggNum).constraints[k].op;
    switch (t) {
      case INT:
        tempint = rule->getAggregate(aggNum).constraints[k].intVal;
        if (o == EQ) {
          if (event->getAttribute(idx).intVal != tempint)
            return false;
        } else if (o == GT) {
          if (event->getAttribute(idx).intVal <= tempint)
            return false;
        } else if (o == LT) {
          if (event->getAttribute(idx).intVal >= tempint)
            return false;
        } else if (o == LE) {
          if (event->getAttribute(idx).intVal > tempint)
            return false;
        } else if (o == GE) {
          if (event->getAttribute(idx).intVal < tempint)
            return false;
        }

        break;

      case FLOAT:
        tempfloat = rule->getAggregate(aggNum).constraints[k].floatVal;
        if (o == EQ) {
          if (event->getAttribute(idx).floatVal != tempfloat)
            return false;
        } else if (o == GT) {
          if (event->getAttribute(idx).floatVal <= tempfloat)
            return false;
        } else if (o == LT) {
          if (event->getAttribute(idx).floatVal >= tempfloat)
            return false;
        } else if (o == LE) {
          if (event->getAttribute(idx).floatVal > tempfloat)
            return false;
        } else if (o == GE) {
          if (event->getAttribute(idx).floatVal < tempfloat)
            return false;
        }

        break;

      case BOOL:
        tempbool = rule->getAggregate(aggNum).constraints[k].boolVal;
        if (o == EQ) {
          if (event->getAttribute(idx).boolVal != tempbool)
            return false;
        }
        if (o == NE) {
          if (event->getAttribute(idx).boolVal == tempbool)
            return false;
        } else {
          // Not defined
          return false;
        }
        break;

      case STRING:
        strcpy(tempString, rule->getAggregate(aggNum).constraints[k].stringVal);
        if (o == EQ) {
          if (strcmp(event->getAttribute(idx).stringVal, tempString) != 0)
            return false;
        } else if (o == NE) {
          if (strcmp(event->getAttribute(idx).stringVal, tempString) == 0)
            return false;
        } else {
          // Not defined
          return false;
        }
        break;
    }
  }
  return true;
}

bool checkNegationConstraints(PubPkt* event, RulePkt* rule, int negNum) {
  int idx, tempint;
  float tempfloat;
  bool tempbool;
  char tempString[STRING_VAL_LEN];
  ValType t;
  Op o;
  for (int k = 0; k < rule->getNegation(negNum).constraintsNum; k++) {
    if (event->getAttributeIndexAndType(
            rule->getNegation(negNum).constraints[k].name, idx, t) == false)
      return false;
    o = rule->getNegation(negNum).constraints[k].op;
    switch (t) {
      case INT:
        tempint = rule->getNegation(negNum).constraints[k].intVal;
        if (o == EQ) {
          if (event->getAttribute(idx).intVal != tempint)
            return false;
        } else if (o == GT) {
          if (event->getAttribute(idx).intVal <= tempint)
            return false;
        } else if (o == LT) {
          if (event->getAttribute(idx).intVal >= tempint)
            return false;
        } else if (o == LE) {
          if (event->getAttribute(idx).intVal > tempint)
            return false;
        } else if (o == GE) {
          if (event->getAttribute(idx).intVal < tempint)
            return false;
        }

        break;

      case FLOAT:
        tempfloat = rule->getNegation(negNum).constraints[k].floatVal;
        if (o == EQ) {
          if (event->getAttribute(idx).floatVal != tempfloat)
            return false;
        } else if (o == GT) {
          if (event->getAttribute(idx).floatVal <= tempfloat)
            return false;
        } else if (o == LT) {
          if (event->getAttribute(idx).floatVal >= tempfloat)
            return false;
        } else if (o == LE) {
          if (event->getAttribute(idx).floatVal > tempfloat)
            return false;
        } else if (o == GE) {
          if (event->getAttribute(idx).floatVal < tempfloat)
            return false;
        }
        break;

      case BOOL:
        tempbool = rule->getNegation(negNum).constraints[k].boolVal;
        if (o == EQ) {
          if (event->getAttribute(idx).boolVal != tempbool)
            return false;
        }
        if (o == NE) {
          if (event->getAttribute(idx).boolVal == tempbool)
            return false;
        } else {
          // Not defined
          return false;
        }
        break;

      case STRING:
        strcpy(tempString, rule->getNegation(negNum).constraints[k].stringVal);
        if (o == EQ) {
          if (strcmp(event->getAttribute(idx).stringVal, tempString) != 0)
            return false;
        } else if (o == NE) {
          if (strcmp(event->getAttribute(idx).stringVal, tempString) == 0)
            return false;
        } else {
          // Not defined
          return false;
        }
        break;
    }
  }
  return true;
}

void getMatchingColumns(PubPkt* e, map<int, set<int>>& topics,
                        list<int>& results) {
  map<int, set<int>>::iterator it = topics.find(e->getEventType());
  if (it == topics.end())
    return;
  for (set<int>::iterator statesIt = it->second.begin();
       statesIt != it->second.end(); ++statesIt) {
    int state = *statesIt;
    results.push_back(state);
  }
}

void getMatchingColumns(int e, map<int, set<int>>& topics, list<int>& results) {
  map<int, set<int>>::iterator it = topics.find(e);
  if (it == topics.end())
    return;
  for (set<int>::iterator statesIt = it->second.begin();
       statesIt != it->second.end(); ++statesIt) {
    int state = *statesIt;
    results.push_back(state);
  }
}

EventInfo createEventInfo(PubPkt* event) {
  EventInfo info;
  if (event->getAttributesNum() > MAX_NUM_ATTR) {
    cout << "You need to raise MAX_NUM_ATTR to " << event->getAttributesNum()
         << " to perform this computation. Exiting" << endl;
    exit(-1);
  }
  for (int i = 0; i < event->getAttributesNum(); i++) {
    info.attr[i].type = event->getAttribute(i).type;
    strcpy(info.attr[i].name, event->getAttribute(i).name);
    switch (event->getAttribute(i).type) {
      case INT:
        info.attr[i].intVal = event->getAttribute(i).intVal;
        break;

      case FLOAT:
        info.attr[i].floatVal = event->getAttribute(i).floatVal;
        break;

      case BOOL:
        info.attr[i].boolVal = event->getAttribute(i).boolVal;
        break;

      case STRING:
        strcpy(info.attr[i].stringVal, event->getAttribute(i).stringVal);
        break;
    }
  }
  info.timestamp = event->getTimeStamp().getTimeVal();
  return info;
}

int getFirstValidElementCircular(uint64_t* column, int lowIndex, int highIndex,
                                 uint64_t minTimeStamp) {
  // minTimeStamp is not included in the selection; Jan 2015
  int size;
  if (highIndex >= lowIndex)
    size = highIndex - lowIndex;
  else
    size = ALLOC_SIZE + highIndex - lowIndex;
  int relativeMinValue = 0;
  int relativeMaxValue = size - 1;
  int minValue = lowIndex;
  int maxValue = highIndex - 1;
  if (maxValue < 0)
    maxValue = ALLOC_SIZE + maxValue;
  if (column[maxValue] <= minTimeStamp)
    return -1;
  while (relativeMaxValue - relativeMinValue > 1) {
    int relativeMidPoint =
        relativeMinValue + (relativeMaxValue - relativeMinValue) / 2;
    int midPoint = (relativeMidPoint + lowIndex) % ALLOC_SIZE;
    if (column[midPoint] <= minTimeStamp) {
      relativeMinValue = relativeMidPoint;
      minValue = midPoint;
    } else {
      relativeMaxValue = relativeMidPoint;
      maxValue = midPoint;
    }
  }
  if (relativeMaxValue - relativeMinValue == 0)
    return minValue;
  if (column[minValue] > minTimeStamp)
    return minValue;
  return maxValue;
}

int getLastValidElementCircular(uint64_t* column, int lowIndex, int highIndex,
                                uint64_t maxTimeStamp, int minIndex) {
  // maxTimeStamp is not included in the selection; Jan 2015
  if (minIndex < 0)
    return -1;
  int size;
  if (highIndex >= lowIndex)
    size = highIndex - lowIndex;
  else
    size = ALLOC_SIZE + highIndex - minIndex;
  int relativeMinValue = 0;
  int relativeMaxValue = size - 1;
  int minValue = lowIndex;
  int maxValue = highIndex - 1;
  if (maxValue < 0)
    maxValue = ALLOC_SIZE - maxValue;
  if (column[minIndex] >= maxTimeStamp) {
    return -1;
  }
  while (relativeMaxValue - relativeMinValue > 1) {
    int relativeMidPoint =
        relativeMinValue + (relativeMaxValue - relativeMinValue) / 2;
    int midPoint = (relativeMidPoint + lowIndex) % ALLOC_SIZE;
    if (column[midPoint] >= maxTimeStamp) {
      relativeMaxValue = relativeMidPoint;
      maxValue = midPoint;
    } else {
      relativeMinValue = relativeMidPoint;
      minValue = midPoint;
    }
  }
  if (relativeMaxValue == relativeMinValue)
    return minValue;
  if (column[maxValue] < maxTimeStamp)
    return maxValue;
  return minValue;
}

int deleteInvalidElementsCircular(uint64_t* column, int& lowIndex,
                                  int highIndex, uint64_t minTimeStamp) {
  int columnSize;
  if (highIndex - lowIndex >= 0)
    columnSize = highIndex - lowIndex;
  else
    columnSize = ALLOC_SIZE + highIndex - lowIndex;
  int firstValidElement =
      getFirstValidElementCircular(column, lowIndex, highIndex, minTimeStamp);
  if (firstValidElement < 0) {
    lowIndex = highIndex;
    return 0;
  }
  if (firstValidElement == lowIndex)
    return columnSize;
  lowIndex = firstValidElement;
  int newSize;
  if (highIndex >= lowIndex)
    newSize = highIndex - lowIndex;
  else
    newSize = ALLOC_SIZE + highIndex - lowIndex;
  return newSize;
}

Parameter* dupParameter(Parameter* param) {
  Parameter* result = new Parameter;
  memcpy(result, param, sizeof(Parameter));
  return result;
}

// END OF GPU CODE IMPORT

int getFirstValidElement(vector<PubPkt*>& column, int columnSize,
                         TimeMs minTimeStamp) {
  // minTimeStamp is not included in the selection; Jan 2015
  if (columnSize <= 0)
    return -1;
  int minValue = 0;
  int maxValue = columnSize - 1;
  if (column[maxValue]->getTimeStamp() <= minTimeStamp)
    return -1;
  while (maxValue - minValue > 1) {
    int midPoint = minValue + (maxValue - minValue) / 2;
    if (column[midPoint]->getTimeStamp() <= minTimeStamp) {
      minValue = midPoint;
    } else {
      maxValue = midPoint;
    }
  }
  if (maxValue - minValue == 0)
    return minValue;
  if (column[minValue]->getTimeStamp() > minTimeStamp)
    return minValue;
  return maxValue;
}

int getLastValidElement(vector<PubPkt*>& column, int columnSize,
                        TimeMs maxTimeStamp, int minIndex) {
  // maxTimeStamp is not included in the selection; Jan 2015
  int minValue = minIndex;
  int maxValue = columnSize - 1;
  if (minIndex == -1)
    return -1;
  if (column[minIndex]->getTimeStamp() >= maxTimeStamp)
    return -1;
  while (maxValue - minValue > 1) {
    int midPoint = minValue + (maxValue - minValue) / 2;
    if (column[midPoint]->getTimeStamp() >= maxTimeStamp) {
      maxValue = midPoint;
    } else {
      minValue = midPoint;
    }
  }
  if (maxValue - minValue == 0)
    return minValue;
  if (column[maxValue]->getTimeStamp() < maxTimeStamp)
    return maxValue;
  return minValue;
}

class CompareTS {
public:
  bool operator()(const TimeMs& a, PubPkt* b) { return a < b->getTimeStamp(); }
  bool operator()(PubPkt* a, const TimeMs& b) { return a->getTimeStamp() < b; }
  bool operator()(const TimeMs& a, const TimeMs& b) { return a < b; }
  bool operator()(PubPkt* a, PubPkt* b) {
    return a->getTimeStamp() < b->getTimeStamp();
  }
};

vector<PubPkt*>::iterator getBeginPacket(vector<PubPkt*>& column,
                                         TimeMs minTime) {
  return std::upper_bound(column.begin(), column.end(), minTime, CompareTS());
}

vector<PubPkt*>::iterator getEndPacket(vector<PubPkt*>& column,
                                       TimeMs maxTime) {
  return std::lower_bound(column.begin(), column.end(), maxTime, CompareTS());
}

int deleteInvalidElements(vector<PubPkt*>& column, int columnSize,
                          TimeMs minTimeStamp) {
  if (columnSize == 0)
    return columnSize;
  if (column[columnSize - 1]->getTimeStamp() == minTimeStamp)
    return 1;
  int firstValidElement =
      getFirstValidElement(column, columnSize, minTimeStamp);
  if (firstValidElement < 0)
    return 0;
  else
    return columnSize - firstValidElement;
}

float computeFloatValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                     OpTree* opTree, int index,
                                     StateType sType) {
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
    int refIndex = pktReference->getIndex();
    PubPkt* cPkt;
    // are we checking the current state packet?
    if ((refIndex == index && pktReference->refersToAgg() == 0 &&
         pktReference->refersToNeg() == 0 && sType == STATE) ||
        (pktReference->refersToNeg() && sType == NEG) ||
        (pktReference->refersToAgg() && sType == AGG)) {
      cPkt = pkt;
    } else {
      cPkt = partialEvent->indexes[refIndex];
    }
    int attrIndex;
    ValType vType;
    if (cPkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                       vType) == false)
      return 0;
    if (vType == INT)
      return cPkt->getIntAttributeVal(attrIndex);
    else if (vType == FLOAT)
      return cPkt->getFloatAttributeVal(attrIndex);
    else if (vType == BOOL)
      return cPkt->getBoolAttributeVal(attrIndex);
  } else {
    // Integer can only be obtained from integer: assume this is ensured at rule
    // deployment time
    float leftValue = computeFloatValueForParameters(
        pkt, partialEvent, opTree->getLeftSubtree(), index, sType);
    float rightValue = computeFloatValueForParameters(
        pkt, partialEvent, opTree->getRightSubtree(), index, sType);

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

int computeIntValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                 OpTree* opTree, int index, StateType sType) {
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
    int refIndex = pktReference->getIndex();
    PubPkt* cPkt;
    // are we checking the current state packet?
    if ((refIndex == index && pktReference->refersToAgg() == 0 &&
         pktReference->refersToNeg() == 0 && sType == STATE) ||
        (pktReference->refersToNeg() && sType == NEG) ||
        (pktReference->refersToAgg() && sType == AGG)) {
      cPkt = pkt;
    } else {
      cPkt = partialEvent->indexes[refIndex];
    }
    int attrIndex;
    ValType vType;
    if (cPkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                       vType) == false)
      return 0;
    if (vType == INT)
      return cPkt->getIntAttributeVal(attrIndex);
    else if (vType == FLOAT)
      return cPkt->getFloatAttributeVal(attrIndex);
    else if (vType == BOOL)
      return cPkt->getBoolAttributeVal(attrIndex);
  } else {
    // Integer can only be obtained from integer: assume this is ensured at rule
    // deployment time
    int leftValue = computeIntValueForParameters(
        pkt, partialEvent, opTree->getLeftSubtree(), index, sType);
    int rightValue = computeIntValueForParameters(
        pkt, partialEvent, opTree->getRightSubtree(), index, sType);
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

bool computeBoolValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                   OpTree* opTree, int index, StateType sType) {
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
    int refIndex = pktReference->getIndex();

    PubPkt* cPkt;
    // are we checking the current state packet?
    if ((refIndex == index && pktReference->refersToAgg() == 0 &&
         pktReference->refersToNeg() == 0 && sType == STATE) ||
        (pktReference->refersToNeg() && sType == NEG) ||
        (pktReference->refersToAgg() && sType == AGG)) {
      cPkt = pkt;
    } else {
      cPkt = partialEvent->indexes[refIndex];
    }
    int attrIndex;
    ValType vType;
    if (cPkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                       vType) == false)
      return false;
    if (vType == BOOL)
      return cPkt->getIntAttributeVal(attrIndex);
    else if (vType == FLOAT)
      return cPkt->getFloatAttributeVal(attrIndex);
    else if (vType == BOOL)
      return cPkt->getBoolAttributeVal(attrIndex);
  } else {
    // Integer can only be obtained from integer: assume this is ensured at rule
    // deployment time
    bool leftValue = computeBoolValueForParameters(
        pkt, partialEvent, opTree->getLeftSubtree(), index, sType);
    bool rightValue = computeBoolValueForParameters(
        pkt, partialEvent, opTree->getRightSubtree(), index, sType);

    OpTreeOperation op = opTree->getOp();
    if (op == AND)
      return leftValue && rightValue;
    if (op == OR)
      return leftValue || rightValue;
  }
  return false;
}

void computeStringValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                     OpTree* opTree, int index, StateType sType,
                                     char* res) {
  OpTreeType type = opTree->getType();
  if (type == LEAF) {
    OpValueReference* reference = opTree->getValueReference();
    RulePktValueReference* pktReference =
        dynamic_cast<RulePktValueReference*>(reference);
    if (pktReference == NULL) {
      StaticValueReference* sReference =
          dynamic_cast<StaticValueReference*>(reference);
      if (sReference->getType() == STRING)
        return sReference->getStringValue(res);
    }
    int refIndex = pktReference->getIndex();

    PubPkt* cPkt;
    // are we checking the current state packet?
    if ((refIndex == index && pktReference->refersToAgg() == 0 &&
         pktReference->refersToNeg() == 0 && sType == STATE) ||
        (pktReference->refersToNeg() && sType == NEG) ||
        (pktReference->refersToAgg() && sType == AGG)) {
      cPkt = pkt;
    } else {
      cPkt = partialEvent->indexes[refIndex];
    }
    int attrIndex;
    ValType vType;
    if (cPkt->getAttributeIndexAndType(pktReference->getName(), attrIndex,
                                       vType) == false) {
      strcpy(res, "");
      return;
    }
    if (vType == STRING)
      return cPkt->getStringAttributeVal(attrIndex, res);
  } else {
    // Binary tree operations are not supported for strings
    return;
  }
}

bool checkComplexParameter(PubPkt* pkt, PartialEvent* partialEvent,
                           CPUParameter* parameter, int index,
                           StateType sType) {
  Op operation = parameter->operation;
  // TODO: what if types don't match?
  ValType type = parameter->vtype;
  if (type == INT) {
    int left = computeIntValueForParameters(pkt, partialEvent,
                                            parameter->leftTree, index, sType);
    int right = computeIntValueForParameters(
        pkt, partialEvent, parameter->rightTree, index, sType);
    if (operation == EQ)
      return left == right;
    if (operation == GT)
      return left > right;
    if (operation == LT)
      return left < right;
    if (operation == NE)
      return left != right;
    if (operation == LE)
      return left <= right;
    if (operation == GE)
      return left >= right;
  } else if (type == FLOAT) {
    float left = computeFloatValueForParameters(
        pkt, partialEvent, parameter->leftTree, index, sType);
    float right = computeFloatValueForParameters(
        pkt, partialEvent, parameter->rightTree, index, sType);
    if (operation == EQ)
      return left == right;
    if (operation == GT)
      return left > right;
    if (operation == LT)
      return left < right;
    if (operation == NE)
      return left != right;
    if (operation == LE)
      return left <= right;
    if (operation == GE)
      return left >= right;
  } else if (type == BOOL) {
    bool leftValue = computeBoolValueForParameters(
        pkt, partialEvent, parameter->leftTree, index, sType);
    bool rightValue = computeBoolValueForParameters(
        pkt, partialEvent, parameter->rightTree, index, sType);
    if (operation == EQ)
      return leftValue && rightValue;
    if (operation == NE)
      return leftValue || rightValue;
  } else if (type == STRING) {
    char strLeft[STRING_VAL_LEN], strRight[STRING_VAL_LEN];
    computeStringValueForParameters(pkt, partialEvent, parameter->leftTree,
                                    index, sType, strLeft);
    computeStringValueForParameters(pkt, partialEvent, parameter->rightTree,
                                    index, sType, strRight);
    if (operation == EQ)
      return (strcmp(strLeft, strRight) == 0);
    if (operation == NE)
      return (strcmp(strLeft, strRight) != 0);
  }
  return false;
}
