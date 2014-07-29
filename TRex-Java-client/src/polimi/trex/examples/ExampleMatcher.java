//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Daniele Rogora
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

package polimi.trex.examples;

import java.util.Date;

import polimi.trex.common.Matcher;
import polimi.trex.packets.PubPkt;

/**
 * This is a simple example matcher; it checks that the packets to filter have been generated on Tuesday
 * @author Daniele Rogora
 *
 */
public class ExampleMatcher implements Matcher {

	@SuppressWarnings("deprecation")
	@Override
	public boolean match(PubPkt packet) {
		Date date = new Date(packet.getTimeStamp());
		//Matches only if the packet was generated on Tuesday
		if (date.getDay()==2) return true;
		else return false;
	}

}
