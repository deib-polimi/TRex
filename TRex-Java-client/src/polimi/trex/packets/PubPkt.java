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

import polimi.trex.common.Attribute;


/**
 * A publication packet. It is used by sources to send events to the T-Rex engine.
 */
public class PubPkt implements TRexPkt {
	private int eventType;
	protected long timeStamp;
	protected Collection<Attribute> attributes;
	
	public PubPkt(int eventType) {
		this.eventType = eventType;
		this.timeStamp = System.currentTimeMillis();
		attributes = new ArrayList<Attribute>();
	}
	
	public PubPkt(int eventType, long timeStamp) {
		this.eventType = eventType;
		this.timeStamp = timeStamp;
		attributes = new ArrayList<Attribute>();
	}

	public int getEventType() {
		return eventType;
	}

	public void setEventType(int eventType) {
		this.eventType = eventType;
	}

	public long getTimeStamp() {
		return timeStamp;
	}

	public void setTimeStamp(long timeStamp) {
		this.timeStamp = timeStamp;
	}

	public Collection<Attribute> getAttributes() {
		return attributes;
	}

	public void addAttribute(Attribute attribute) {
		attributes.add(attribute);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof PubPkt)) return false;
		PubPkt other = (PubPkt) obj;
		if (eventType != other.eventType) return false;
		if (timeStamp != other.timeStamp) return false;
		if (! attributes.containsAll(other.attributes)) return false;
		if (! other.attributes.containsAll(attributes)) return false;
		return true;
	}
}