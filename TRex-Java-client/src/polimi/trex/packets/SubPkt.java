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

package polimi.trex.packets;

import java.util.ArrayList;
import java.util.Collection;

import polimi.trex.common.Constraint;


public class SubPkt implements TRexPkt {
	private int eventType;
	protected Collection<Constraint> constraints;
	
	public SubPkt(int eventType) {
		this.eventType = eventType;
		constraints = new ArrayList<Constraint>();
	}
	
	public int getEventType() {
		return eventType;
	}

	public void setEventType(int eventType) {
		this.eventType = eventType;
	}

	public Collection<Constraint> getConstraints() {
		return constraints;
	}

	public void addConstraint(Constraint constraint) {
		constraints.add(constraint);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof SubPkt)) return false;
		SubPkt other = (SubPkt) obj;
		if (eventType != other.eventType) return false;
		if (! constraints.containsAll(other.constraints)) return false;
		if (! other.constraints.containsAll(constraints)) return false;
		return true;
	}
}
