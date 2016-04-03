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

#ifndef GENERICROUTINGTABLE_H_
#define GENERICROUTINGTABLE_H_

#include <set>
#include "../Packets/PubPkt.h"
#include "../Packets/SubPkt.h"

/**
 * This represents a generic routing table.
 * Implementing classes have to define the behaviors for:
 * 1) Installing and removing subscriptions
 * 2) Compute the set of clients matching a a given publication
 */
class GenericRoutingTable {

public:
  virtual ~GenericRoutingTable() {}

  /**
   * Installs a new subscription for the given client.
   */
  virtual void installSubscription(int clientId, SubPkt* subscription) = 0;

  /**
   * Deletes the subscription for the given client.
   */
  virtual void deleteSubscription(int clientId, SubPkt* subscription) = 0;

  /**
   * Deletes alla subscriptions stored for the given client.
   */
  virtual void removeClient(int clientId) = 0;

  /**
   * Returns the set of clients matching the given publication.
   * Results are stored in the clients parameter.
   */
  virtual void getMatchingClients(PubPkt* pubPkt, std::set<int>& clients) = 0;
};

#endif /* GENERICROUTINGTABLE_H_ */
