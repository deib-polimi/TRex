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

#ifndef INDEXINGTABLE_H_
#define INDEXINGTABLE_H_

#include "IndexingTableCommon.h"
#include "NoConstraintIndex.h"
#include "IntConstraintIndex.h"
#include "FloatConstraintIndex.h"
#include "BoolConstraintIndex.h"
#include "StringConstraintIndex.h"
#include "../Packets/RulePkt.h"

/**
 * An IndexingTable indexes automaton states, making it more efficient to identify
 * states that may be interested in a message.
 * The table indexes positive states as well as negations. It contains a different
 * indexing structure for each type of value allowed in messages; this means that
 * adding a new type needs the definition of a new indexing structure.
 *
 * The indexing table only considers "static" constraints, i.e. constraints that only
 * depend on the information present in the state at deploy time. Timing constraints
 * and parameters (which depend on the history of messages received) are not indexed
 * since they would need a high indexing cost at runtime.
 */
class IndexingTable {
public:

	IndexingTable();

	virtual ~IndexingTable();

	/**
	 * Indexes the information about the states used in the given automaton
	 */
	void installRulePkt(RulePkt *pkt);

	/**
	 * Fills mh with the set of all matching states
	 */
	void processMessage(PubPkt *pkt, MatchingHandler &mh);

private:
	std::map<int, IntConstraintIndex> intIndex;				// EventType -> Index for constraints on integer values
	std::map<int, FloatConstraintIndex> floatIndex;		// EventType -> Index for constraints on float values
	std::map<int, BoolConstraintIndex> boolIndex;			// EventType -> Index for constraints on boolean values
	std::map<int, StringConstraintIndex> stringIndex;	// EventType -> Index for constraints on string values
	std::map<int, NoConstraintIndex> noIndex;					// EventType -> Index of "no" constraints

	/**
	 * Predicates installed in tables
	 */
	std::set<TablePred *> usedPreds;

	/**
	 * Installs the given TablePred with the specified eventType and constraints.
	 * It uses the two functions below to install event type and constraints.
	 */
	inline void installTablePredicate(int eventType, Constraint *constraint, int numConstraints, TablePred *tp);
	inline void installNoConstraint(int eventType, TablePred *tp);
	inline void installConstraint(int eventType, Constraint &constr, TablePred *tp);

};

#endif
