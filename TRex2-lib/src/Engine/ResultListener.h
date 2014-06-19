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

#ifndef RESULTLISTENER_H_
#define RESULTLISTENER_H_

#include <set>
#include "../Packets/PubPkt.h"

/**
 * A ResultListener can be connected to the processing engine to receive results.
 * It is an abstract class: extending subclasses must define the handleResult method
 * to actually define the result processing behavior.
 */
class ResultListener {

public:

	virtual ~ResultListener() { }

	/**
	 * Receives results from the processing engine.
	 * The genPkts parameter is the set of generated packets.
	 * The meanProcTime represents the mean time for processing a message, in microseconds.
	 *
	 * Important: at the end of this function messages will be automatically delete.
	 * The developer MUST call the incRefCount function if it wants to store the packet locally.
	 */
	virtual void handleResult(std::set<PubPkt *> &genPkts, double procTime) = 0;

};

#endif
