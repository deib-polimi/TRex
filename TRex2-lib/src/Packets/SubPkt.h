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

#ifndef SUBPKT_H_
#define SUBPKT_H_

#include "../Common/Consts.h"
#include "../Common/Structures.h"

/**
 * This class represents a subscription packet, which is composed by a type,
 * and a set of constraints (name-operator-value triples).
 */
class SubPkt {
public:

	/**
	 * Constructor.
	 * Parameters:
	 * parEventType: eventType.
	 * Builds a new subscription packet without constraints.
	 */
	SubPkt(int parEventType);

	/**
	 * Constructor.
	 * Parameters:
	 * parEventType: event type
	 * parConstraints: array of constraints
	 * parConstraintsNum: size of the parConstraints array
	 */
	SubPkt(int parEventType, Constraint *parConstraints, int parConstraintsNum);

	/**
	 * Copy constructor
	 */
	SubPkt(const SubPkt &pkt);

	/**
	 * Destructor
	 */
	virtual ~SubPkt();

	/**
	 * Getter methods
	 */
	int getEventType() { return eventType; }

	int getConstraintsNum() { return constraintsNum; }

	Constraint getConstraint(int constraintsNum) { return constraints[constraintsNum]; }

private:

	int eventType;						// Type of the event
	Constraint *constraints;	// Set of constraints
	int constraintsNum;				// Number of attributes
};

#endif /* SUBPKT_H_ */
