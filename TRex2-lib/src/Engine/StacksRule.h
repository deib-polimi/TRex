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

#ifndef STACKSRULE_H_
#define STACKSRULE_H_

#include "../Common/Funs.h"
#include "../Common/Consts.h"
#include "../Common/TimeMs.h"
#include "../Packets/PubPkt.h"
#include "../Packets/RulePkt.h"
#include "../Engine/IndexingTableCommon.h"
#include "CompositeEventGenerator.h"
#include "Stack.h"
#include <list>
#include <map>
#include <vector>

/**
 * Represents a single detection sequence: it is represented by a set of stacks
 * with optional negations and parameters.
 */
class StacksRule {
public:
  /**
   * Constructor: sets the ruleID and build stacks
   * from RulePkt passed as parameter
   */
  StacksRule(RulePkt* pkt);

  /**
   * Destructor
   */
  virtual ~StacksRule();

  /**
   * Returns the rule id
   */
  int getRuleId() { return ruleId; }

  /**
   * Adds the received packet to the aggregate with the given index
   */
  void addToAggregateStack(PubPkt* pkt, int index);

  /**
   * Adds the received packet to the negation with the given index
   */
  void addToNegationStack(PubPkt* pkt, int index);

  /**
   * Adds the packet to the given index.
   * Index must be different from 0
   */
  void addToStack(PubPkt* pkt, int index);

  /**
   * Adds the given packet to stack 0 and starts
   * the computation of composite events.
   */
  void startComputation(PubPkt* pkt, std::set<PubPkt*>& results);

  /**
   * Process pkt: used only for testing purpose
   */
  void processPkt(PubPkt* pkt, MatchingHandler* mh, std::set<PubPkt*>& results,
                  int index);

private:
  // The id of the rule
  int ruleId;
  RulePkt* rulePkt;

  // Stacks in the rule (stack id -> data structure)
  std::vector<Stack> stacks;
  // Set of parameters to check at the end
  std::vector<Parameter> endStackParameters;
  // Parameters in the rule to check in the meantime
  std::map<int, std::vector<CPUParameter>> branchStackComplexParameters;
  // Parameters in the rule to check in the meantime
  // (stack id -> data structure)
  // std::map<int, std::set<Parameter *> > branchStackParameters;
  // Parameters in the rule to check in the meantime
  // (negation id -> data structure)
  // std::map<int, std::set<Parameter *> > negationParameters;

  // Parameters in the rule to check in the meantime
  // (negation id -> data structure)
  std::map<int, std::vector<CPUParameter>> negationComplexParameters;
  // Parameters in the rule to check in the meantime
  // (aggregate id -> data structure)
  std::map<int, std::vector<CPUParameter>> aggregateComplexParameters;
  // Aggregate in the rule (aggregate id -> data structure)
  std::vector<Aggregate> aggregates;
  // Negations in the rule (negation id -> data structure)
  std::vector<Negation> negations;
  // Number of stacks in the rule
  int stacksNum;
  // Number of aggregates in the rule
  int aggrsNum;
  // Number of negations in the rule
  int negsNum;

  // Stack id -> state it refers to in the rule
  std::vector<int> referenceState;

  // Number of pkts stored for each stack in the rule
  std::vector<int> stacksSize;
  // Number of pkts stored for each negation in the rule
  std::vector<int> negsSize;
  // Number of pkts stored for each aggregate in the sequence
  std::vector<int> aggsSize;

  // Aggregate index -> set of all matching PubPkt
  std::vector<std::vector<PubPkt*>> receivedAggs;
  // Stack index -> set of all matching PubPkt
  std::vector<std::vector<PubPkt*>> receivedPkts;
  // Negation index -> set of all matching PubPkt
  std::vector<std::vector<PubPkt*>> receivedNegs;

  // Indexes of events in the consuming clause (set of stack ids)
  std::set<int> consumingIndexes;
  // Used to generate composite event attributes (if any)
  CompositeEventGenerator eventGenerator;
  // Used to generate a composite event id no attributes are defined
  int compositeEventId;

  /**
   * Adds the packet to the given stack (can be a normal stack, or a stack for
   * negations or aggregates)
   */
  inline void parametricAddToStack(PubPkt* pkt, int& parStacksSize,
                                   std::vector<PubPkt*>& parReceived);

  /**
   * Adds a new parameter constraint
   */
  inline void addParameter(int index1, char* name1, int index2, char* name2,
                           StateType type, RulePkt* pkt);

  inline void addComplexParameter(Op pOperation, OpTree* lTree, OpTree* rTree,
                                  int lastIdx, StateType type, ValType vtype);

  int computeIntValue(PubPkt* pkt, PartialEvent& partialEvent, OpTree* opTree,
                      int index, bool isNeg);

  /**
   * Adds a new negation to negations map
   */
  inline void addNegation(int eventType, Constraint constraints[],
                          int constrLen, int lowIndex, TimeMs& lowTime,
                          int highIndex);

  /**
   * Adds a new aggregate to aggregates map
   */
  inline void addAggregate(int eventType, Constraint constraints[],
                           int constrLen, int lowIndex, TimeMs& lowTime,
                           int highIndex, char* name, AggregateFun& fun);

  /**
   * Returns the events that satisfy the stack window with the given from the
   * given time stamp
   */
  inline void getWinEvents(std::list<PartialEvent>& partialEvents, int index,
                           TimeMs tsUp, CompKind mode,
                           PartialEvent& partialEvent);

  /**
   * Return true for the presence of negation (according to parameters)
   */
  inline bool checkNegation(int negIndex, PartialEvent& partialResult);

  /**
   * Computes partial results and returns them as a list of PartialEvent.
   */
  inline std::list<PartialEvent> getPartialResults(PubPkt* pkt);

  /**
   * Computes complex events and adds them to the results set
   */
  inline void createComplexEvents(std::list<PartialEvent>& partialEvents,
                                  std::set<PubPkt*>& results);

  /**
   * Removes events that have been consumed
   */
  inline void removeConsumedEvent(std::list<PartialEvent>& partialEvents);

  /**
   * Returns true if the parameter is satisfied by the packet
   */
  inline bool checkParameter(PubPkt* pkt, PartialEvent& partialEvent,
                             Parameter& parameter);

  // inline bool checkComplexParameter(PubPkt *pkt, PartialEvent
  // *partialEvent, CPUParameter *parameter, int index, bool isNeg);

  /**
   * Returns true if all parameters are satisfied by the packet
   */
  inline bool checkParameters(PubPkt* pkt, PartialEvent& partialEvent,
                              std::vector<CPUParameter>& complexParameters,
                              int index, StateType sType);

  /**
   * Removes partial events that do not match parameters
   */
  inline void removePartialEventsNotMatchingParameters(
      std::list<PartialEvent>& partialEvents,
      std::vector<Parameter>& parameters);

  /**
   * Remove packets invalidated by timing constraints.
   */
  inline void clearStacks();

  /**
   * Removes all packets that are older than minTS from the given stack.
   * The stack can be a normal stack, or a stack for negations or aggregates.
   */
  inline void removeOldPacketsFromStack(TimeMs& minTS, int& parStacksSize,
                                        std::vector<PubPkt*>& parReceived);
};

#endif
