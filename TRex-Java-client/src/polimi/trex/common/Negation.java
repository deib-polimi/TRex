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

package polimi.trex.common;

import java.util.ArrayList;
import java.util.Collection;

/**
 * Defines a negation, i.e. a constraint that imposes an event not to happen in a given interval.
 * It is used inside a RulePkt.
 */
public class Negation {
	private int eventType;
	private Collection<Constraint> constraints;
	private int lowerId;			// If lowerId<0, the negation is time based
	private long lowerTime;		// Considered only if lowerId<0
	private int upperId;
	
	private Negation(int eventType) {
		this.eventType = eventType;
		constraints = new ArrayList<Constraint>();
	}
	
	/**
	 * Constructor defining a negation between two events. 
	 */
	public Negation(int eventType, int lowerId, int upperId) {
		this(eventType);
		this.lowerId = lowerId;
		this.upperId = upperId;
		this.lowerTime = 0;
	}
	
	/**
	 * Constructor defining a time based negation. 
	 */
	public Negation(int eventType, int upperId, long lowerTime) {
		this(eventType, -1, upperId);
		this.lowerTime = lowerTime;
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

	public int getLowerId() {
		return lowerId;
	}

	public void setLowerId(int lowerId) {
		this.lowerId = lowerId;
	}

	public long getLowerTime() {
		return lowerTime;
	}

	public void setLowerTime(long lowerTime) {
		this.lowerTime = lowerTime;
	}

	public int getUpperId() {
		return upperId;
	}

	public void setUpperId(int upperId) {
		this.upperId = upperId;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof Negation)) return false;
		Negation other = (Negation) obj;
		if (constraints.size() != other.constraints.size()) return false;
		if (! constraints.containsAll(other.constraints)) return false;
		if (lowerId != other.lowerId) return false;
		if (lowerTime != other.lowerTime) return false;
		if (upperId != other.upperId) return false;
		return true;
	}
}
