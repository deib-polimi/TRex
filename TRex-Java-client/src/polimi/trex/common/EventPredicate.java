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

import java.util.Collection;

import polimi.trex.common.Consts.CompKind;


/**
 * Represents a basic event predicate. 
 */
public class EventPredicate {
	private int eventType;
	private Collection<Constraint> constraints;
	private int refersTo;
	private long win;
	private CompKind kind;
	
	/** Creates the root predicate */
	public EventPredicate(int eventType, Collection<Constraint> constraints) {
		this(eventType, constraints, -1, 0, CompKind.EACH_WITHIN);
	}
	
	public EventPredicate(int eventType, Collection<Constraint> constraints, int refersTo, long win, CompKind kind) {
		this.eventType = eventType;
		this.constraints = constraints;
		this.refersTo = refersTo;
		this.win = win;
		this.kind = kind;
	}

	public int getEventType() {
		return eventType;
	}

	public Collection<Constraint> getConstraints() {
		return constraints;
	}

	public int getRefersTo() {
		return refersTo;
	}

	public long getWin() {
		return win;
	}

	public CompKind getKind() {
		return kind;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof EventPredicate)) return false;
		EventPredicate other = (EventPredicate) obj;
		if (! constraints.containsAll(other.constraints)) return false;
		if (! other.constraints.containsAll(constraints)) return false;
		if (eventType != other.eventType) return false;
		if (kind != other.kind) return false;
		if (refersTo != other.refersTo) return false;
		if (win != other.win) return false;
		return true;
	}	
}
