//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Francesco Feltrinelli
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

/**
 * Identifies a T-Rex Pkt. All packets used to communicate with T-Rex should implement it. 
 */
public interface TRexPkt { 
	
	public enum PktType {
		PUB_PKT,
		RULE_PKT,
		SUB_PKT,
		ADV_PKT,
		JOIN_PKT,
	}
	
	public enum PacketType {
		PUB_PACKET(PktType.PUB_PKT.ordinal()),
		SUB_PACKET(PktType.SUB_PKT.ordinal()),
		RULE_PACKET(PktType.RULE_PKT.ordinal()),
		UNSUB_PACKET(100),
		PING_PACKET(101);
		
		private final int value;
		PacketType(int value) { this.value = value; }

	    public int toValue() { return value; }

	    /**
	     * Get a {@link PacketType} from its value.
	     * @param value the {@link PacketType}'s value
	     * @return the corresponding {@link PacketType}
	     * @throws IllegalArgumentException if no match is found
	     */
	    public static PacketType fromValue(int value){
	    	for (PacketType packetType: values()){
	    		if (packetType.value == value) return packetType;
	    	}
	    	throw new IllegalArgumentException("There is no packet with value "+value);
	    }
	}
	
}
