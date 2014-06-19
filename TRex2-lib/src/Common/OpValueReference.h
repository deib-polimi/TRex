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

#ifndef OPVALUEREFERENCE_H_
#define OPVALUEREFERENCE_H_

#include "Consts.h"

/**
 * This interface represents a reference to an actual value, used inside an OpTree.
 * By re-defining the inner data structure in concrete subclasses, it allows the
 * OpTree structure to be used in different contexts.
 */
class OpValueReference {
public:

	virtual ~OpValueReference() { }

	/**
	 * Creates an exact copy (deep copy) of the data structure
	 */
	virtual OpValueReference * dup() = 0;
	
	/**
	 * This describes the type of the reference
	 */
	ValRefType vrtype;
};

#endif /* OPVALUEREFERENCE_H_ */
