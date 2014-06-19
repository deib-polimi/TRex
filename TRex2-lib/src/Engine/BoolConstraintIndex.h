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

#ifndef BOOLCONSTRAINTINDEX_H_
#define BOOLCONSTRAINTINDEX_H_

#include "AbstractConstraintIndex.h"
#include "../Packets/PubPkt.h"

/**
 * Represents a boolean constraint stored in the index table
 */
typedef struct BoolTableConstraintStruct {
	char name[NAME_LEN];												// Attribute name
	Op op;																			// Operator
	bool val;																		// Attribute value
	std::set<TablePred *> connectedPredicates;	// Set of predicates using the constraint
} BoolTableConstraint;

/**
 * Contains a Value -> Constraint index for each defined operator
 */
typedef struct BoolOperatorsTable {
	std::map<bool, BoolTableConstraint *> eq;	// Value -> equality constraint
	std::map<bool, BoolTableConstraint *> df;	// Value -> different from constraint
	// Overriding
	bool operator<(const BoolOperatorsTable &table) const { return eq<table.eq; }
} BoolOps;

/**
 * Represents the index for constraints of boolean values.
 */
class BoolConstraintIndex : AbstractConstraintIndex {
public:

	BoolConstraintIndex();

	/**
	 * Frees dynamic memory
	 */
	virtual ~BoolConstraintIndex();

	/**
	 * Creates or gets the BoolTableConstraint C representing the constraint
	 * given as parameter. Then it installs the predicate in C.
	 */
	void installConstraint(Constraint &constraints, TablePred *predicate);

	/**
	 * Processes the given message, using the partial results stored in predCount.
	 * It updates predCount and fills mh with the matching states.
	 */
	void processMessage(PubPkt *pkt, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

private:

	std::map<std::string, BoolOps> indexes;						// Name -> indexes for that name
	std::set<BoolTableConstraint *> usedConstraints;	// Set of all constraints used in the table

	/**
	 * Checks if there already exists an BoolTableConstraints which is
	 * compatible with the constraint c.
	 * If it finds a valid BoolTableConstraints, return a pointer to it,
	 * otherwise returns null.
	 */
	BoolTableConstraint * getConstraint(Constraint &c);

	/**
	 * Creates a new BoolTableConstraint using the information stored in the parameter constraint
	 */
	BoolTableConstraint * createConstraint(Constraint &c);

	/**
	 * Installs the given constraint to the appropriate table
	 */
	inline void installConstraint(BoolTableConstraint *c);

	/**
	 * Processes the given constraint by updating the predCount and, if needed, the mh structures
	 */
	inline void processConstraint(BoolTableConstraint *c, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

};

#endif
