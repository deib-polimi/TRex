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

#ifndef ABSTRACTCONSTRAINTINDEX_H_
#define ABSTRACTCONSTRAINTINDEX_H_

#include "IndexingTableCommon.h"

/**
 * Represents a generic constraint index.
 * It includes all common functions used by constraint indexes specialized for a given type.
 */
class AbstractConstraintIndex {

protected:

	/**
	 * Adds the information of tp in mh
	 */
	void addToMatchingHandler(MatchingHandler &mh, TablePred *tp);

};

#endif
