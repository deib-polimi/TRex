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

#ifndef INTCONSTRAINTINDEX_H_
#define INTCONSTRAINTINDEX_H_

#include "AbstractConstraintIndex.h"
#include "../Packets/PubPkt.h"

/**
 * Represents an integer constraint stored in the index table
 */
typedef struct IntTableConstraintStruct {
	char name[NAME_LEN];													// Attribute name
	Op op;																				// Operator
	int val;																			// Attribute value
	std::set<TablePred *> connectedPredicates;		// Set of predicates using the constraint
} IntTableConstraint;

/**
 * Contains a Value -> Constraint index for each defined operator
 */
typedef struct IntOperatorsTable {
	std::map<int, IntTableConstraint *> eq;	// Value -> equality constraint
	std::map<int, IntTableConstraint *> lt;	// Value -> less than constraint
	std::map<int, IntTableConstraint *> gt;	// Value -> greater then constraint
	std::map<int, IntTableConstraint *> df;	// Value -> different from constraint
	// Overriding
	bool operator<(const IntOperatorsTable &table) const { return eq<table.eq; }
} IntOps;

/**
 * Represents the index for constraints of integer values.
 */
class IntConstraintIndex : AbstractConstraintIndex {
public:

	IntConstraintIndex();

	/**
	 * Frees dynamic memory
	 */
	virtual ~IntConstraintIndex();

	/**
	 * Creates or gets the IntTableConstraint C representing the constraint
	 * given as parameter. Then it installs the predicate in C.
	 */
	void installConstraint(Constraint &constraints, TablePred *predicate);

	/**
	 * Processes the given message, using the partial results stored in predCount.
	 * It updates predCount and fills mh with the matching states.
	 */
	void processMessage(PubPkt *pkt, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

private:

	std::map<std::string, IntOps> indexes;					// Name -> indexes for that name
	std::set<IntTableConstraint *> usedConstraints;	// Set of all constraints used in the table

	/**
	 * Checks if there already exists an IntTableConstraints which is
	 * compatible with the constraint c.
	 * If it finds a valid IntTableConstraints, return a pointer to it,
	 * otherwise returns null.
	 */
	IntTableConstraint * getConstraint(Constraint &c);

	/**
	 * Creates a new IntTableConstraint using the information stored in the parameter constraint
	 */
	IntTableConstraint * createConstraint(Constraint &c);

	/**
	 * Installs the given constraint to the appropriate table
	 */
	inline void installConstraint(IntTableConstraint *c);

	/**
	 * Processes the given constraint by updating the predCount and, if needed, the mh structures
	 */
	inline void processConstraint(IntTableConstraint *c, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

};

#endif
