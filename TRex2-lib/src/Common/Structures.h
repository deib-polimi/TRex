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

#ifndef STRUCTURES_H_
#define STRUCTURES_H_

#include "Consts.h"
#include "OpTree.h"

/**
 * A simple attribute
 */
typedef struct EventAttribute {
  char name[NAME_LEN + 1];
  ValType type;
  union {
    int intVal;
    float floatVal;
    bool boolVal;
    char stringVal[STRING_VAL_LEN + 1];
  };
} Attribute;

/**
 * A simple constraint
 */
typedef struct EventConstraint {
  char name[NAME_LEN + 1];
  Op op;
  ValType type;
  int intVal;
  float floatVal;
  bool boolVal;
  char stringVal[STRING_VAL_LEN + 1];

  bool operator==(const EventConstraint& x) const {
    if (op != x.op)
      return false;
    if (type != x.type)
      return false;
    if (strcmp(name, x.name) != 0)
      return false;
    if (type == INT && intVal != x.intVal)
      return false;
    if (type == FLOAT && floatVal != x.floatVal)
      return false;
    if (type == BOOL && boolVal != x.boolVal)
      return false;
    if (type == STRING && strcmp(stringVal, x.stringVal) != 0)
      return false;
    return true;
  }
} Constraint;

/**
 * Represents a parameter constraint.
 * It requires the value of name1 in predicate evIndex1
 * to be equal to the value of name2 in predicate evIndex2.
 */
typedef struct ParameterConstraints {
  // Index of the first state involved in the constraint
  int evIndex1;
  // Name of the first attribute involved in the constraint
  char name1[NAME_LEN + 1];
  // Index of the second state involved in the constraint
  int evIndex2;
  // Name of the second attribute involved in the constraint
  char name2[NAME_LEN + 1];
  // Decides whether the second index refers to a normal state,
  // to an aggregate, or to a negation
  StateType type;
  ValType vtype;

  int firstAttr;
  int secondAttr;
} Parameter;

typedef struct ComplexParameterStruct {
  Op operation;
  // Decides whether the parameter should be used during the computation of a
  // normal state, to an aggregate, or to a negation
  StateType type;
  ValType vtype;
  OpTree* leftTree;
  OpTree* rightTree;
} ComplexParameter;

/**
 * Represents a complex parameter for the CPU engine. It asks operation to be
 * verified between the values (of type vtype) computed analyzing the trees
 */
typedef struct ComplexParameterConstraints {
  Op operation;
  // Decides whether the parameter should be used during the computation of a
  // normal state, to an aggregate, or to a negation
  StateType type;
  ValType vtype;
  OpTree* leftTree;
  OpTree* rightTree;
  int lastIndex;
} CPUParameter;

/**
 * The representation of a node of a binary tree for the GPU serialized array;
 * it can be both a static value or a reference to an attribute of an event
 */
typedef struct GPUNode {
  OpTreeOperation operation;
  OpTreeType type;
  StateType sType;
  int refersTo;
  int attrNum;
  bool empty;
  bool isStatic;
  char attrName[NAME_LEN + 1];
  ValType valueType;
  union {
    int intVal;
    float floatVal;
    bool boolVal;
    char stringVal[STRING_VAL_LEN + 1];
  };
} Node;

/**
 * Represents a complex parameter for the GPU engine. It asks operation to be
 * verified between the values (of type vtype) computed analyzing the trees
 */
typedef struct ComplexGPUParameterConstraints {
  Op operation;
  // Decides whether the second index refers to a normal state,
  // to an aggregate, or to a negation
  StateType sType;
  ValType vType;
  Node leftTree[TREE_SIZE];
  int lSize;
  int rSize;
  Node rightTree[TREE_SIZE];
  int lastIndex;
  int negIndex;
  int aggIndex;
  int depth;
} GPUParameter;

/**
 * Represents a negation. It means: no event of type eventId between constraints
 * identified by lowerId and upperId
 */
typedef struct GNeg {
  // Id of the lower bound constraint (-1 if the lower bound is time based)
  int lowerId;
  // Time of the lower bound
  uint64_t lowerTime;
  // Id of the upper bound constraint
  int upperId;
} GPUNegation;

/**
 * Represents a negation. It means: no event of type eventId between constraints
 * identified by lowerId and upperId
 */
typedef struct Neg {
  // Type of the event
  int eventType;
  // Constraints on event content
  Constraint* constraints;
  // Number of constraints
  int constraintsNum;
  // Id of the lower bound constraint (-1 if the lower bound is time based)
  int lowerId;
  // Time of the lower bound
  TimeMs lowerTime;
  // Id of the upper bound constraint
  int upperId;
} Negation;

/**
 * Represents an aggregate.
 * It means: take all event of type eventId between constraints identified by
 * lowerId and upperId and compute the given function to the values of the
 * attribute with the given name.
 */
typedef struct Agg {
  // Type of the event
  int eventType;
  // Constraints on event content
  Constraint* constraints;
  // Number of constraints
  int constraintsNum;
  // Id of the lower bound constraint (-1 if the lower bound is time based)
  int lowerId;
  // Time of the lower bound
  TimeMs lowerTime;
  // Id of the upper bound constraint
  int upperId;
  // Aggregate function
  AggregateFun fun;
  // Name of the attribute to use for computation
  char name[NAME_LEN + 1];
} Aggregate;

/**
 * Represents a parameter constraint shared between more than one sequence
 */
typedef struct ExtParam {
  int seqId1;
  int evIndex1;
  char name1[NAME_LEN + 1];
  int seqId2;
  int evIndex2;
  char name2[NAME_LEN + 1];
} ExtParameter;

typedef struct EventInfoStruct {
  Attribute attr[MAX_NUM_ATTR];
  uint64_t timestamp;
  uint32_t uid;
  bool operator==(const EventInfoStruct& x) const {
    if (timestamp != x.timestamp)
      return false;
    for (int i = 0; i < MAX_NUM_ATTR; i++) {
      if (attr[i].intVal != x.attr[i].intVal)
        return false;
    }
    return true;
  }

} EventInfo;

typedef struct IntEventInfoSetStruct {
  EventInfo infos[MAX_RULE_FIELDS];
} EventInfoSet;

typedef struct {
  int workspaceNum;
  // Results computed during the evaluation of the current column
  int* resPtr;
  int* resultsSizePtr;
  // Pointers to the events information in device memory
  int eventsChunks[MAX_RULE_FIELDS - 1][MAX_PAGES_PER_STACK];
  int numEventChunks[MAX_RULE_FIELDS];
  int aggregatesChunks[MAX_NUM_AGGREGATES][MAX_PAGES_PER_STACK];
  // Pointers to the events information in device memory stored for aggregates
  int numAggregatesChunks[MAX_NUM_AGGREGATES];
  int negationsChunks[MAX_NEGATIONS_NUM][MAX_PAGES_PER_STACK];
  int numNegationsChunks[MAX_NEGATIONS_NUM];
  // Pointers to partial values for aggregates in device memory
  int* aggregatesValPtr;
  GPUParameter* parametersPtr;
  // Pointers to the parameters defined for each aggregate in device memory
  GPUParameter* aggregatesParametersPtr;
  GPUParameter* negationsParametersPtr;
  GPUNegation* negationsPtr;
} ruleMemoryStruct;

typedef struct {
  int inUse;
  int onGpu;
  int pageIdx;
} chunk;

typedef struct {
  // The index of the parameter to be analyzed
  int paramNum[MAX_PARAMETERS_NUM];
} parPasser;

typedef struct {
  // The index of the page to be used.
  // Allows [evt_index,col,rule -> memory page] mapping
  int pageIdx[MAX_PAGES_PER_STACK];
} passer;

typedef struct {
  int size[MAX_NEGATIONS_NUM];
  int minId[MAX_NEGATIONS_NUM];
  int pages[MAX_NEGATIONS_NUM][MAX_PAGES_PER_STACK];
  int parameters[MAX_NEGATIONS_NUM][MAX_PARAMETERS_NUM];
} negsPasser;

typedef struct {
  // Results computed during the evaluation of the previous column
  EventInfoSet* prevResultsPtr;
  EventInfoSet* currentResultsPtr;
  uint8_t* alivePtr;
  int* seqMaxPtr;
  float* aggregatesValPtr;
} workspace;

#endif
