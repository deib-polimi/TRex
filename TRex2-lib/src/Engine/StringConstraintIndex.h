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

#ifndef STRINGCONSTRAINTINDEX_H_
#define STRINGCONSTRAINTINDEX_H_

#include "AbstractConstraintIndex.h"
#include "../Packets/PubPkt.h"

/**
 * Represents a string constraint stored in the index table
 */
typedef struct StringTableConstraintStruct {
	char name[NAME_LEN];												// Attribute name
	Op op;																			// Operator
	std::string val							;								// Attribute value
	std::set<TablePred *> connectedPredicates;	// Set of predicates using the constraint
} StringTableConstraint;

/**
 * Contains a Value -> Constraint index for each defined operator
 */
typedef struct StringOperatorsTable {
	std::map<std::string, StringTableConstraint *> eq;	// Value -> equality constraint
	std::map<std::string, StringTableConstraint *> in;	// Value -> include contraint
	// Overriding
	bool operator<(const StringOperatorsTable &table) const { return eq<table.eq; }
} StringOps;

/**
 * Represents the index for constraints of integer values.
 */
class StringConstraintIndex : AbstractConstraintIndex {
public:

	StringConstraintIndex();

	/**
	 * Frees dynamic memory
	 */
	virtual ~StringConstraintIndex();

	/**
	 * Creates or gets the StringTableConstraint C representing the constraint
	 * given as parameter. Then it installs the predicate in C.
	 */
	void installConstraint(Constraint &constraints, TablePred *predicate);

	/**
	 * Processes the given message, using the partial results stored in predCount.
	 * It updates predCount and fills mh with the matching states.
	 */
	void processMessage(PubPkt *pkt, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

private:

	std::map<std::string, StringOps> indexes;						// Name -> indexes for that name
	std::set<StringTableConstraint *> usedConstraints;	// Set of all constraints used in the table

	/**
	 * Checks if there already exists an StringTableConstraints which is
	 * compatible with the constraint c.
	 * If it finds a valid StringTableConstraints, return a pointer to it,
	 * otherwise returns null.
	 */
	StringTableConstraint * getConstraint(Constraint &c);

	/**
	 * Creates a new StringTableConstraint using the information stored in the parameter constraint
	 */
	StringTableConstraint * createConstraint(Constraint &c);

	/**
	 * Installs the given constraint to the appropriate table
	 */
	inline void installConstraint(StringTableConstraint *c);

	/**
	 * Processes the given constraint by updating the predCount and, if needed, the mh structures
	 */
	inline void processConstraint(StringTableConstraint *c, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

};

#endif
