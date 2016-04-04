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

#ifndef RULEPKT_H_
#define RULEPKT_H_

#include "../Common/Consts.h"
#include "../Common/Structures.h"
#include "../Common/TimeMs.h"
#include "../Common/CompositeEventTemplate.h"
#include "../Packets/RulePktValueReference.h"
#include "../Packets/StaticValueReference.h"
#include <set>
#include <vector>

/**
 * A basic event predicate
 */
typedef struct EventPredicate {
  // Type of the event required by this predicate
  int eventType;
  // Predicate constraints
  Constraint constraints[];
  // Number of constraints in the predicate
  int constraintsNum;
  // Index of the reference predicate (-1 if root)
  int refersTo;
  // Detection time window
  TimeMs win;
  // The kind of constraint
  CompKind kind;
} Predicate;

/**
 * A RulePkt contains the definition of a composite event.
 * More in particular, it contains the pattern that must be detected
 * and the template of the composite event to generate.
 * The pattern is expressed through a set of predicates, a set of negations,
 * a set of aggregates, and a set of parameters.
 */
class RulePkt {
public:
  // #if MP_MODE == MP_COPY
  /**
   * Creates an exact copy of the packet
   */
  RulePkt* copy();
  //#endif

  RulePkt(bool resetCount);

  virtual ~RulePkt();

  /**
   * Adds the root predicate.
   * Returns false if an error occurs
   */
  bool addRootPredicate(int eventId, Constraint constraints[], int constrLen);

  /**
   * Adds a predicate; root predicate must be already defined.
   * Returns false if an error occurs.
   */
  bool addPredicate(int eventType, Constraint constraints[], int constrLen,
                    int refersTo, const TimeMs& win, CompKind kind);

  /**
   * Adds a new time based negation.
   * It asks no events of type eventType satisfying all constraints to occur
   * within win from the state with id referenceId.
   * Returns false if an error occurs.
   */
  bool addTimeBasedNegation(int eventType, Constraint constraints[],
                            int constrLen, int referenceId, const TimeMs& win);

  /**
   * Adds a negation between two states.
   * It asks no events of type eventType satisfying all constraints to occur
   * between the states with ids id1 and id2.
   * Returns false if an error occurs.
   */
  bool addNegationBetweenStates(int eventType, Constraint constraints[],
                                int constrLen, int id1, int id2);

  /**
   * Adds a new parameter between states.
   * It asks the value of the attribute having name name1 in the event used for
   * id id1 to be equal to the value of the attribute having name name2 in the
   * event used for id id2.
   */
  bool addParameterBetweenStates(int id1, char* name1, int id2, char* name2);

  /**
   * Add a new parameter between states
   * It asks that the comparison pOperation is satisfied when checked between
   * the leftTree and the rightTree; these trees may involve any state of the
   * sequence or static value
   */
  bool addComplexParameter(Op pOperation, ValType type, OpTree* leftTree,
                           OpTree* rightTree);

  /**
   * Adds a new parameter that is analyzed when searching for a possible
   * negation event
   * It asks that the comparison pOperation is satisfied when checked between
   * the leftTree and the rightTree; these trees may NOT involve any state of
   * the sequence (read below). They can also refer to static values and 1, and
   * only one, negation.
   * IMPORTANT: for this to work as expected on the GPU engine the parameter
   * MUST NOT refer to any state that comes later than the last state to which
   * the negation refers in the sequence
   */
  bool addComplexParameterForNegation(Op pOperation, ValType type,
                                      OpTree* leftTree, OpTree* rightTree);

  /**
   * Add a new parameter between that is analyzed when computing an aggregate.
   * It asks that the comparison pOperation is satisfied when checked between
   * the leftTree and the rightTree; these trees may involve any state of the
   * sequence or static value and the aggregate event that is being analyzed
   */
  bool addComplexParameterForAggregate(Op pOperation, ValType type,
                                       OpTree* leftTree, OpTree* rightTree);

  /**
   * Adds a new parameter between a state and a negation.
   * It asks the value of the attribute having name name in the event used for
   * id to be equal to the value of the attribute having name negName in the
   * event used for negation negId.
   */
  bool addParameterForNegation(int id, char* name, int negId, char* negName);

  /**
   * Adds a new parameter between a state and an aggregate.
   * It asks the value of the attribute having name name in the event used for
   * id to be equal to the value of the attribute having name aggName in the
   * event used for aggregate aggId.
   */
  bool addParameterForAggregate(int id, char* name, int aggId, char* aggName);

  /**
   * Adds a new time based aggregate to compute the given function fun using a
   * set of values.
   * In particular, it uses all the values of attributes having the given name
   * coming from events having the given eventType and satisfying all
   * constraints, occurring within win from the event used for the referenceId.
   */
  bool addTimeBasedAggregate(int eventType, Constraint constraints[],
                             int constrLen, int referenceId, const TimeMs& win,
                             char* name, AggregateFun fun);

  /**
   * Adds a new aggregate to compute the given function fun using a set of
   * values.
   * In particular, it uses all the values of attributes having the given name
   * coming from events having the given eventType and satisfying all
   * constraints, occurring within the events used for the ids id1 and id2.
   */
  bool addAggregateBetweenStates(int eventType, Constraint constraints[],
                                 int constrLen, int id1, int id2, char* name,
                                 AggregateFun fun);

  /**
   * Adds a consuming clause for the given eventIndex.
   * Returns false if an error occurs.
   */
  bool addConsuming(int eventIndex);

  /**
   * Sets the template for the composite event generated by the rule
   */
  void setCompositeEventTemplate(CompositeEventTemplate* parCeTemplate) {
    ceTemplate = parCeTemplate;
  }

  /**
   * Fills leaves with the set of indexes that are leaves in the ordering graph
   */
  std::set<int> getLeaves() const;

  /**
   * Fills joinPoints with the set of indexes that are shared
   * among more than one sequence
   */
  std::set<int> getJoinPoints() const;

  /**
   * Returns the number of predicates in the rule
   */
  int getPredicatesNum() const { return predicates.size(); }

  /**
   * Returns the number of parameters in the rule
   */
  int getParametersNum() const { return parameters.size(); }

  // Is always the same as GPUComplexParameters, no need for another method
  int getComplexParametersNum() const { return complexParameters.size(); }

  /**
   * Returns the number of negations in the rule
   */
  int getNegationsNum() const { return negations.size(); }

  /**
   * Returns the number of aggregates in the rule
   */
  int getAggregatesNum() const { return aggregates.size(); }

  /**
   * Returns the predicate with the given index
   */
  Predicate& getPredicate(int index) const { return predicates[index]; }

  /**
   * Returns the parameter with the given index
   */
  Parameter& getParameter(int index) const { return parameters[index]; }

  /**
   * Returns the complex parameter for the CPU with the given index
   */
  CPUParameter& getComplexParameter(int index) const {
    return complexParameters[index];
  }

  /**
   * Returns the complex parameter for the GPU with the given index
   */
  GPUParameter& getComplexGPUParameter(int index) const {
    return complexGPUParameters[index];
  }

  /**
   * Returns the negation with the given index
   */
  Negation& getNegation(int index) const { return negations[index]; }

  /**
   * Returns the aggregate with the given index
   */
  Aggregate& getAggregate(int index) const { return aggregates[index]; }

  /**
   * Returns the set of consuming indexes
   */
  std::set<int> getConsuming() const { return consuming; }

  /**
   * Returns the composite event template
   */
  CompositeEventTemplate* getCompositeEventTemplate() const {
    return ceTemplate;
  }

  /**
   * Returns the rule id
   */
  int getRuleId() const { return ruleId; }

  /**
   * Returns true if the subscription contains at least
   * a predicate with the given eventType
   */
  bool containsEventType(int eventType, bool includeNegations) const;

  /**
   * Returns the set of all contained event types
   */
  std::set<int> getContainedEventTypes() const;

  /**
   * Get the maximum time window between the two events
   * Requires lowerId < upperId
   * && lowerId < getConstraintNum()
   * && upperId < getConstraintNum()
   */
  TimeMs getWinBetween(int lowerId, int upperId) const;

  /**
   * Returns the maximum length of a sequence defined in the subscription
   */
  TimeMs getMaxWin() const;

  /**
   * Returns true if id1 is directly defined through id2, or viceversa
   */
  bool isDirectlyConnected(int id1, int id2) const;

  /**
   * Returns true if id1 is defined through id2, or viceversa
   */
  bool isInTheSameSequence(int id1, int id2) const;

  /**
   * Overloading
   */
  bool operator<(const RulePkt& pkt) const;
  bool operator==(const RulePkt& pkt) const;
  bool operator!=(const RulePkt& pkt) const;

private:
  // Last used rule identifier
  static int lastId;
  // Identifier of the rule
  int ruleId;
  // Array of event predicates
  std::vector<Predicate> predicates;
  // Parameters between different predicates (map identifier -> data structure)
  std::vector<Parameter> parameters;
  // Negations defined for the rule (map identifier -> data structure)
  std::vector<Negation> negations;
  // Aggregates defined for the rule (map identifier -> data structure)
  std::vector<Aggregate> aggregates;
  // Set of indexes of events that are consumed after detection
  std::set<int> consuming;
  std::vector<CPUParameter> complexParameters;
  std::vector<GPUParameter> complexGPUParameters;
  // Template of the composite event
  CompositeEventTemplate* ceTemplate;

  /**
   * Adds a new negation negation.
   * Predicates identified by lowerId and upperId must be already defined.
   * If lowerId < 0, then the lowerTime bound is used.
   * Returns false if an error occurs
   */
  bool addNegation(int eventType, Constraint constraints[], int constrLen,
                   int lowerId, const TimeMs& lowerTime, int upperId);

  /**
   * Adds the parameter with the given attributes
   * The type determines whether the second index refers to a normal state,
   * to a negation, or to an aggregate.
   */
  bool addParameter(int index1, char* name1, int index2, char* name2,
                    StateType type);

  /**
   * Adds a new aggregate.
   * Predicates identified by lowerId and upperId must be already defined.
   * If lowerId < 0, then the lowerTime bound is used.
   * Returns false if an error occurs
   */
  bool addAggregate(int eventType, Constraint constraints[], int constrLen,
                    int lowerId, const TimeMs& lowerTime, int upperId,
                    char* name, AggregateFun fun);

  /**
   * Fills referenceCount with the number of times each index is referenced
   */
  std::map<int, int> getReferenceCount() const;

  /**
   * Returns the index of the last state in a given tree
   */
  int findLastState(OpTree* tree) const;

  /**
   * Returns the index of the unique negation in the given tree
   */
  int findNegation(OpTree* tree) const;

  /**
   * Returns the index of the unique aggregate in the given tree
   */
  int findAggregate(OpTree* tree) const;

  /**
   * Returns the depth of the given tree
   */
  int findDepth(OpTree* tree, int depth) const;

  /**
   * Serializes the trees passed as parameters and writes the results in the
   * given GPUComplexParameter structure
   */
  void serializeTrees(OpTree* ltree, OpTree* rtree, int depth,
                      GPUParameter& gp) const;

  /**
   * Does the actual job for the serializeTrees function;
   * operates on one tree at a time
   */
  void serializeNode(OpTree* tree, int& idx, Node serialized[]) const;
};

#endif
