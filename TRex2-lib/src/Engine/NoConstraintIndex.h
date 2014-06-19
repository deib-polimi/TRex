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

#ifndef NOCONSTRAINTINDEX_H_
#define NOCONSTRAINTINDEX_H_

#include "AbstractConstraintIndex.h"
#include "../Packets/PubPkt.h"

/**
 * Represents the index for predicated without constraints of integer values.
 */
class NoConstraintIndex : AbstractConstraintIndex {
public:

	NoConstraintIndex();

	/**
	 * Frees dynamic memory
	 */
	virtual ~NoConstraintIndex();

	/**
	 * Creates or gets the IntTableConstraint C representing the constraint
	 * given as parameter. Then it installs the predicate in C.
	 */
	void installPredicate(TablePred *predicate);

	/**
	 * Processes the given message, using the partial results stored in predCount.
	 * It updates predCount and fills mh with the matching states.
	 */
	void processMessage(PubPkt *pkt, MatchingHandler &mh, std::map<TablePred *, int> &predCount);

private:

	std::set<TablePred *> predicates;
};

#endif
