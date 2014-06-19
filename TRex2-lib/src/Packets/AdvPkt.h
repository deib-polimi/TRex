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

#ifndef ADVPKT_H_
#define ADVPKT_H_

#include <set>

/**
 * This class represents an advertisement packet, which is composed by a set of types.
 */
class AdvPkt {

public:

	/**
	 * Constructor with parameters
	 */
	AdvPkt(std::set<int> parAdvertisements);

	/**
	 * Copy constructor
	 */
	AdvPkt(const AdvPkt &pkt);

	/**
	 * Destructor
	 */
	virtual ~AdvPkt();

	/**
	 * Getter methods
	 */
	std::set<int> getAdvertisements() { return advertisements; }

private:

	std::set<int> advertisements;	// Set of advertised event types
};

#endif /* ADVPKT_H_ */
