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

#ifndef GPU_PROCESSOR_H
#define GPU_PROCESSOR_H

#include <set>
#include <cuda_runtime.h>
#include "../../Common/Consts.h"
#include "../../Common/Funs.h"
#include "../../Packets/RulePkt.h"
#include "CudaKernels.h"
#include "MemoryManager.h"
#include "GPUProcessorIf.h"

using namespace std;

class GPUProcessor : public GPUProcessorIf {
public:
  /**
   * Constructor
   */
  GPUProcessor(RulePkt* rulePkt, MemoryManager* m);

  /**
   * Destructor
   */
  ~GPUProcessor();

  /**
   * Initializes all data structure on device memory.
   * This is kept separate from the constructor because it must be called
   * by the same cpu thread invoking the processEvent method.
   */
  void init();

  /**
   * Processes the given event.
   */
  void processEvent(PubPkt* event, set<PubPkt*>& generatedEvents);

  /**
   * Returns the minimum element in the given stack having
   * a time stamp greater than minTimestamp.
   * If agg==true, stack represents an aggregate stack.
   */
  int getMinValidElement(int stack, uint64_t minTimestamp, StateType t);

  /**
   * Returns the maximum element in the given stack having
   * a time stamp greater than maxTimestamp.
   * If agg==true, stack represents an aggregate stack.
   */
  int getMaxValidElement(int stack, uint64_t maxTimestamp, StateType t);

  /**
   * Returns the low index for the given stack.
   * If agg==true, stack represents an aggregate stack.
   */
  int getLowIndex(int stack, StateType t);

  /**
   * Returns the high index for the given stack.
   * If agg==true, stack represents an aggregate stack.
   */
  int getHighIndex(int stack, StateType t);

  /**
   * Returns the time stamp of the event stored in the given stack,
   * at the given index.
   * If agg==true, stack represents an aggregate stack.
   */
  uint64_t getTimestamp(int stack, int index, StateType t);

  /**
   * Returns the CompKind for the given state
   */
  CompKind getCompKind(int state);

  /**
   * Returns the id of the compositeEvent associated with the rule
   */
  int getCompositeEventId();

  /**
   * Returns the number of attributes belonging
   * to the compositeEvent associated with the rule
   */
  int getAttributesNum();

  /**
   * Returns the window of the predicate state
   */
  uint64_t getWin(int state);

  /**
   * Returns the number of predicates of the rule
   */
  int getSequenceLen();

  /**
   * Returns the number of aggregates of the rule
   */
  int getAggregatesNum();

  /**
   * Fills idx, o and val with the informations about
   * the constraint num of the rule state
   */
  void getConstraint(int state, int num, int& idx, Op& o, int& val);

  /**
   * Returns the maximum timestamp associated to the stack stack of kind t
   */
  uint64_t getMaxTimestamp(int stack, StateType t);

  /**
   * Fills index and type with the index and type of the name attribute of the
   * generic event of stack state of type stype
   * Returns true if it succeded, false otherwise
   * (e.g. no attribute with that name or no packets to check)
   */
  bool getAttributeIndexAndType(int state, char* name, int& index,
                                ValType& type, StateType stype);

  /**
   * Returns the number of STATE parameters of the rule
   */
  int getNumParam();

  /**
   * Returns true if the rule consumes events of state state, false otherwise
   */
  bool isConsuming(int state);

  /**
   * Removes ev from the STATE stack stack, if ev exists
   */
  void remove(EventInfo ev, int stack);

  /**
   * Returns the number of negations of the rule
   */
  int getNegationsNum();

  /**
   * Returns a pointer to the Negation num
   */
  Negation* getNegation(int num);

  /**
   * Returns the size of the NEG stack negIndex
   */
  int getNegsSize(int negIndex);

  /**
   * Returns the full CompositeEventTemplate of the rule
   */
  CompositeEventTemplate* getCompositeEvent();

  /**
   * Returns the number of AGG parameters of the rule
   */
  int getNumParamAggregates();

  /**
   * Returns the number of NEG parameters of the rule
   */
  int getNumParamNegations();

  /**
   * Returns the idx of the state referred by predicate state
   */
  int getRefersTo(int state);
  int getStaticAttributesNum();

private:
  bool checkDefinitions();
  bool first;
  int stateParamtersNum;
  int aggregateParametersNum;
  int negationsParametersNum;
  MemoryManager* mm;
  int mmToken;
  /**
   * Adds a new negation to negations map
   */
  inline void addNegation(int eventType, Constraint* constraints, int constrLen,
                          int lowIndex, TimeMs& lowTime, int highIndex);
  // Negations in the rule (negation id -> data structure)
  std::map<int, Negation*> negations;

  // Number of negations in the rule
  int negsNum, aggsNum;
  // Number of pkts stored for each negation in the rule
  int negsSize[MAX_RULE_FIELDS];
  // Negation index -> set of all matching PubPkt
  std::map<int, std::vector<PubPkt*>> receivedNegs;
  bool compareEvt(EventInfo e1, EventInfo e2, int size);
  // Indexes of events in the consuming clause (set of stack ids)
  std::set<int> consumingIndexes;
  map<int, PubPkt*> samples;
  map<int, PubPkt*> aggSamples;
  map<int, PubPkt*> negSamples;
  CompositeEventTemplate* ceTemplate;
  int totevents;
  int sequenceLen;
  int compositeEventId;
  int eventTopics[MAX_RULE_FIELDS];
  CompKind compKinds[MAX_RULE_FIELDS];
  int windows[MAX_RULE_FIELDS];
  // Starting indexes for each stack
  int* lowIndexes;
  // Ending indexes for each stack
  int* highIndexes;
  // Starting indexes for each stack of aggregates
  int* lowIndexesAggs;
  int* highIndexesAggs;
  // Starting indexes for each stack of aggregates
  int* lowIndexesNegs;
  // Ending indexes for each stack of aggregates
  int* highIndexesNegs;
  // Matrix of time stamps
  uint64_t timestamps[MAX_RULE_FIELDS][ALLOC_SIZE];
  // Matrix of time stamps for aggregates stacks
  uint64_t aggTimestamps[MAX_NUM_AGGREGATES][ALLOC_SIZE];
  // Matrix of time stamps for aggregates stacks
  uint64_t negTimestamps[MAX_NEGATIONS_NUM][ALLOC_SIZE];
  EventInfo* negationsInfo[MAX_NEGATIONS_NUM];
  // Cuda kernels
  CudaKernels* kernels;
  // Sequence states -> set of parameters defined for it (as first state)
  map<int, set<GPUParameter*>> parameters;
  // Aggregate id -> aggregate
  map<int, Aggregate*> aggregates;
  // Aggregate id -> set of parameters defined for it
  map<int, set<GPUParameter*>> aggregatesParameters;
  // Aggregate id -> set of parameters defined for it
  map<int, set<GPUParameter*>> negationsParameters;
  // Topic -> set of states with that topic
  map<int, set<int>> topics;
  // Topic -> set of aggregates for that topic
  map<int, set<int>> aggregatesTopics;
  map<int, set<int>> negationsTopics;
  // Rule packet
  RulePkt* rulePkt;

  /**
   * Adds the given parameter to the processing rule
   */
  void addParameter(GPUParameter* parameterp);

  /**
   * Adds the event to the given stack.
   */
  void addEventToStack(int stack, PubPkt* event);

  /**
   * Adds the event to the given stack for aggregates.
   */
  void addEventToAggregateStack(int stack, PubPkt* event);

  void addEventToNegationStack(int stack, PubPkt* event);

  /**
   * Deletes all events that have expired at the event arrival time
   * Returns false if at least a column reaches 0 size
   */
  bool deleteExpiredEvents(PubPkt* event);
};

#endif
