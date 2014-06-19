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

#ifndef SIMPLEROUTINGTABLE_H_
#define SIMPLEROUTINGTABLE_H_

#include "GenericRoutingTable.h"
#include "../Packets/PubPkt.h"
#include <map>
#include <list>
#include <pthread.h>

/**
 * This class represents a simple routing table. It stores subscriptions coming
 * from clients and uses them to compute recipients of events.
 * As the name suggests, it implements a trivial processing approach that iterates
 * through all stored subscriptions.
 * This implementation is thread-safe.
 */
class SimpleRoutingTable : public GenericRoutingTable {

public:
	SimpleRoutingTable();

	virtual ~SimpleRoutingTable();

	void installSubscription(int clientId, SubPkt *subscription);

	void deleteSubscription(int clientId, SubPkt *subscription);

	void removeClient(int clientId);

	void getMatchingClients(PubPkt *pubPkt, std::set<int> &clients);

private:
	std::map<int, std::map<int, std::list<SubPkt *> > > subscriptions;		// Event Type -> Client -> Subscriptions
	pthread_mutex_t *mutex;																								// Mutex to make the access to the class thread-safe

	/**
	 * Return true if and only if s1 and s2 represent the same subscription
	 */
	inline bool sameSubscription(SubPkt *s1, SubPkt *s2);

	/**
	 * Return true if and only if pubPkt matches subPkt.
	 */
	inline bool matches(PubPkt *pubPkt, SubPkt *subPkt);

};

#endif /* SIMPLEROUTINGTABLE_H_ */
