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

#ifndef RULEPKTVALUEREFERENCE_H_
#define RULEPKTVALUEREFERENCE_H_

#include <stdlib.h>
#include <string.h>
#include "../Common/OpValueReference.h"
#include "../Common/Consts.h"

/**
 * A RulePktValueReference extends the OpValueReference class and defines
 * each reference to value by pointing to an attribute or aggregate in the RulePkt.
 */
class RulePktValueReference: public OpValueReference {

public:

	/**
	 * Constructor for a reference to a normal state attribute.
	 * Parameters: index in the packet, and name of the attribute.
	 */
	RulePktValueReference(int stateIndex, char *parAttrName, StateType sType);

	/**
	 * Constructor for a reference to an aggregate state.
	 */
	RulePktValueReference(int aggregateIndex);

	/**
	 * Destructor
	 */
	virtual ~RulePktValueReference();

	/**
	 * Creates an exact copy of the data structure
	 */
	OpValueReference * dup();

	/**
	 * Returns the index of the state in the RulePkt
	 */
	int getIndex();

	/**
	 * Returns the type of reference (to aggregate, or to normal state)
	 */
	bool refersToAgg();
	
	bool refersToNeg();

	/**
	 * Returns the name of the attribute if the object refers to a normal state index.
	 * Returns NULL if the the object refers to an aggregate index.
	 */
	char * getName();
	
	StateType getSType();

private:
	int index;				// Index in the RulePkt
	StateType type;		// Type of the index: pointing to a normal state, or to an aggregate?
	char *attrName;		// Name of the attribute
};

#endif /* RULEPKTVALUEREFERENCE_H_ */
