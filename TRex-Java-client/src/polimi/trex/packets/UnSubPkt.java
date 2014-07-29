//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Francesco Feltrinelli
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

package polimi.trex.packets;

import polimi.trex.packets.TRexPkt;


public class UnSubPkt implements TRexPkt{

	private SubPkt subPkt;

	public UnSubPkt(SubPkt subPkt) {
		super();
		this.subPkt = subPkt;
	}

	public SubPkt getSubPkt() {
		return subPkt;
	}
	
	public int getEventType(){
		return subPkt.getEventType();
	}

	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (!(o instanceof UnSubPkt)) return false;
		UnSubPkt other= (UnSubPkt) o;
		return subPkt.equals(other.subPkt);
	}
}
