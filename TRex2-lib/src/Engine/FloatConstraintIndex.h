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

#ifndef FLOATCONSTRAINTINDEX_H_
#define FLOATCONSTRAINTINDEX_H_

#include "AbstractConstraintIndex.h"
#include "../Packets/PubPkt.h"

/**
 * Represents a float constraint stored in the index table
 */
typedef struct FloatTableConstraintStruct {
	char name[NAME_LEN];												// Attribute name
	Op op;																			// Operator
	float val;																	// Attribute value
	std::set<TablePred *> connectedPredicates;	// Set of predicates using the constraint
} FloatTableConstraint;

/**
 * Contains a Value -> Constraint index for each defined operator
 */
typedef struct FloatOperatorsTable {
	std::map<float, FloatTableConstraint *> eq;	// Value -> equality constraint
	std::map<float, FloatTableConstraint *> lt;	// Value -> less than constraint
	std::map<float, FloatTableConstraint *> gt;	// Value -> greater then constraint
	std::map<float, FloatTableConstraint *> df;	// Value -> different from constraint
	// Overriding
	bool operator<(const FloatOperatorsTable &table) const { return eq<table.eq; }
} FloatOps;

/**
 * Represents the index for constraints of float values.
 */
class FloatConstraintIndex : AbstractConstraintIndex {
public:

	FloatConstraintIndex();

	/**
	 * Frees dynamic memory
	 */
	virtual ~FloatConstraintIndex();

	/**
	 * Creates or gets the FloatTableConstraint C representing the constraint
	 * given as parameter. Then it installs the predicate in C.
	 */
	void installConstraint(Constraint &constraints, TablePred *predicate);

	/**
	 * Processes the given message, using the partial results stored in predCount.
	 * It updates predCount and fills mh with the matching states.
	 */
	void processMessage(PubPkt *pkt, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

private:

	std::map<std::string, FloatOps> indexes;					// Name -> indexes for that name
	std::set<FloatTableConstraint *> usedConstraints;	// Set of all constraints used in the table

	/**
	 * Checks if there already exists an FloatTableConstraints which is
	 * compatible with the constraint c.
	 * If it finds a valid FloatTableConstraints, return a pointer to it,
	 * otherwise returns null.
	 */
	FloatTableConstraint * getConstraint(Constraint &c);

	/**
	 * Creates a new FloatTableConstraint using the information stored in the parameter constraint
	 */
	FloatTableConstraint * createConstraint(Constraint &c);

	/**
	 * Installs the given constraint to the appropriate table
	 */
	inline void installConstraint(FloatTableConstraint *c);

	/**
	 * Processes the given constraint by updating the predCount and, if needed, the mh structures
	 */
	inline void processConstraint(FloatTableConstraint *c, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

};

#endif
