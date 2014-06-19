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

#ifndef INDEXINGTABLECOMMON_H_
#define INDEXINGTABLECOMMON_H_

#include "../Common/Consts.h"
#include <map>
#include <set>

/**
 * Represents a predicate stored in the index table
 */
typedef struct TablePredicateStruct {
	int ruleId; 					// Id of the sequence in the automaton
	int stateId;					// Index of the state in the sequence
	int constraintsNum;		// Number of constraints in the predicate
	StateType stateType;	// State, negation, or aggregate
} TablePred;

/**
 * A matching handler is passed to the indexing table when processing a message.
 * It is filled with the set of matching states, aggregates, and negations.
 */
typedef struct MatchingHandlerStruct {
	std::map<int, std::set<int> > matchingStates;
	std::map<int, std::set<int> > matchingAggregates;
	std::map<int, std::set<int> > matchingNegations;
} MatchingHandler;

#endif
