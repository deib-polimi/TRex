//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara
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

#ifndef FUNS_H_
#define FUNS_H_

#include <stdlib.h>
#include <string.h>
#include <vector>
#include "../Engine/Stack.h"
#include "../Common/TimeMs.h"
#include "../Packets/PubPkt.h"
#include "../Packets/RulePkt.h"
#include <map>
#include <set>
#include <list>

using namespace std;

/**
 * Partial sequence of events, as used during processing
 */
typedef struct PartialEventStruct {
  PubPkt* indexes[MAX_RULE_FIELDS];
} PartialEvent;

bool checkNegationConstraints(PubPkt* event, RulePkt* rule, int negNum);
bool checkAggregateConstraints(PubPkt* event, RulePkt* rule, int aggNum);
bool checkConstraints(PubPkt* event, RulePkt* rule, int state);

/**
 * This file contains all the common functions used during event computation.
 */

/**
 * Returns all the columns matching the incoming event e.
 * Columns indexes are added to the given results list.
 */
void getMatchingColumns(PubPkt* e, map<int, set<int>>& topics,
                        list<int>& results);
void getMatchingColumns(int e, map<int, set<int>>& topics, list<int>& results);

/**
 * Returns the id of the first element in the given column having a
 * value greater than minTimeStamp.
 * Works with a circular buffer, whose elements are defined between
 * lowIndex and highIndex.
 * Returns -1 if such an element cannot be found.
 * The search is performed in logaritmic time, using a binary search.
 */
int getFirstValidElementCircular(uint64_t* column, int lowIndex, int highIndex,
                                 uint64_t minTimeStamp);

/**
 * Returns the id of the last element in the given column having a
 * value smaller than maxTimeStamp and an index greater than minIndex.
 * Works with a circular buffer, whose elements are defined between
 * lowIndex and highIndex.
 * Returns -1 if such an element cannot be found.
 * The search is performed in logaritmic time, using a binary search.
 */
int getLastValidElementCircular(uint64_t* column, int lowIndex, int highIndex,
                                uint64_t maxTimeStamp, int minIndex);

/**
 * Deletes all elements having a value greater or equal than minTimeStamp
 * and moves remaining elements, so that they start from the first position.
 * Works with a circular array and deletion is performed by modifying the given
 * lowIndex.
 * Returns the new size of the column.
 */
int deleteInvalidElementsCircular(uint64_t* column, int& lowIndex,
                                  int highIndex, uint64_t minTimeStamp);

/**
 * Creates a new event info starting from the given event
 */
EventInfo createEventInfo(PubPkt* event);

/**
 * Creates an exact copy of the given parameter
 */
Parameter* dupParameter(Parameter* param);

/**
 * Returns the id of the first element in the given column having a
 * value greater than minTimeStamp.
 * Returns -1 if such an element cannot be found.
 * The search is performed in logarithmic time, using a binary search.
 */
int getFirstValidElement(std::vector<PubPkt*>& column, int columnSize,
                         TimeMs minTimeStamp);

/**
 * Returns the id of the last element in the given column having a
 * value smaller than maxTimeStamp and an index greater than minIndex.
 * Returns -1 if such an element cannot be found.
 * The search is performed in logarithmic time, using a binary search.
 */
int getLastValidElement(std::vector<PubPkt*>& column, int columnSize,
                        TimeMs maxTimeStamp, int minIndex);

/**
 * Returns the new size of the column.
 */
int deleteInvalidElements(std::vector<PubPkt*>& column, int columnSize,
                          TimeMs minTimeStamp);

float computeFloatValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                     OpTree* opTree, int index,
                                     StateType sType);
int computeIntValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                 OpTree* opTree, int index, StateType sType);
bool computeBoolValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                   OpTree* opTree, int index, StateType sType);
bool checkComplexParameter(PubPkt* pkt, PartialEvent* partialEvent,
                           CPUParameter* parameter, int index, StateType sType);
void computeStringValueForParameters(PubPkt* pkt, PartialEvent* partialEvent,
                                     OpTree* opTree, int index, StateType sType,
                                     char* res);

#endif /* FUNS_H_ */
