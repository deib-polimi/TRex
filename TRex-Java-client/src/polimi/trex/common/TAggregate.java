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

import polimi.trex.common.Consts.AggregateFun;


/**
 * Defines an aggregate, i.e. a function computed on a set of values.
 * It is used inside a RulePkt to specify which events should be stored for the computation.
 */
public class TAggregate {
	private int eventType;

	private Collection<Constraint> constraints;
	private int lowerId;				// lowerId<0 represents a time-based aggregate
	private long lowerTime;			// Considered only if lowerId<0
	private int upperId;
	private AggregateFun fun;
	private String name;
	
	private TAggregate(int eventType, int upperId, AggregateFun fun, String name) {
		this.eventType = eventType;
		this.constraints = new ArrayList<Constraint>();
		this.upperId = upperId;
		this.fun = fun;
		this.name = name;
		this.lowerId = -1;
		this.lowerTime = 0;
	}
	
	public TAggregate(int eventType, int lowerId, int upperId, AggregateFun fun, String name) {
		this(eventType, upperId, fun, name);
		this.lowerId = lowerId;
	}
	
	public TAggregate(int eventType, long lowerTime, int upperId, AggregateFun fun, String name) {
		this(eventType, upperId, fun, name);
		this.lowerId = -1;
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

	public AggregateFun getFun() {
		return fun;
	}

	public void setFun(AggregateFun fun) {
		this.fun = fun;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (!(obj instanceof TAggregate)) return false;
		TAggregate other = (TAggregate) obj;
		if (constraints.size()!=other.constraints.size()) return false;
		if (! constraints.containsAll(other.constraints)) return false;
		if (eventType != other.eventType) return false;
		if (fun != other.fun) return false;
		if (lowerId != other.lowerId) return false;
		if (lowerTime != other.lowerTime) return false;
		if (! name.equals(other.name)) return false;
		if (upperId != other.upperId) return false;
		return true;
	}
}
