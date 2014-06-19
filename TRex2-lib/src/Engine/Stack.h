//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Alberto Negrello
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

#ifndef STACK_H_
#define STACK_H_

#include "../Common/Consts.h"
#include <set>

class Stack {
public:

	/**
	 * Constructor
	 */
	Stack(int refersTo, TimeMs window, CompKind kind);

	/**
	 * Destructor
	 */
	virtual ~Stack();

	CompKind getKind() const {
		return kind;
	}

	TimeMs getWindow() const {
		return win;
	}

	int getRefersTo() const {
		return refersTo;
	}

	/**
	 * Add a new referred Stack
	 */
	void addLookBackTo(int reference) {
		lookBackTo->insert(reference);
	}

	/**
	 * Get referred Stacks
	 */
	std::set<int> * getLookBackTo() {
		return lookBackTo;
	}

	/**
	 * Add a new referred Negation
	 */
	void addLinkedNegation(int reference) {
		linkedNegations->insert(reference);
	}

	/**
	 * Get referred Negations
	 */
	std::set<int> *getLinkedNegations() {
		return linkedNegations;
	}

private:
	int refersTo;												// The stack it refers to
	TimeMs win; 												// The maximum time window to look back in the previous column
	CompKind kind; 											// The kind of composition
	std::set<int> *lookBackTo; 					// Referred stacks
	std::set<int> *linkedNegations; 		// Referred negations
};

#endif
