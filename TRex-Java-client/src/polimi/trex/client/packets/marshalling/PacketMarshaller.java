/*
 * Copyright (C) 2011 Francesco Feltrinelli <first_name DOT last_name AT gmail DOT com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

package polimi.trex.client.packets.marshalling;

import polimi.trex.client.packets.Packet;
import polimi.trex.client.packets.PingPacket;
import polimi.trex.client.packets.PubPacket;
import polimi.trex.client.packets.RulePacket;
import polimi.trex.client.packets.SubPacket;
import polimi.trex.client.packets.UnSubPacket;
import polimi.trex.client.packets.Packet.PacketType;
import polimi.trex.common.Consts.EngineType;
import polimi.trex.marshalling.Marshaller;
import polimi.trex.packets.SubPkt;

/**
 * A {@link Packet}'s marshaller. This flattens any kind of {@link Packet} into a
 * byte array. The type of the packet and its length are written as a preamble,
 * followed by its body.
 * 
 * @author Francesco Feltrinelli
 *
 */
public class PacketMarshaller {
	
	/**
	 * Number of bytes used to store the packet type.
	 */
	public final static int BYTENUM_PKTTYPE= 1;
	/**
	 * Number of bytes used to store the packet length.
	 */
	public final static int BYTENUM_PKTLENGTH= 4;
	
	private static class MyMarshaller extends Marshaller {
		
		public static byte[] getByteArray(UnSubPacket pkt) {
			int bodyLen= getNumBytes(pkt);
			byte[] dest = new byte[BYTENUM_PKTTYPE + BYTENUM_PKTLENGTH + bodyLen];
			
			int startIndex= 0;
			startIndex= encode(PacketType.UNSUB_PACKET, dest, startIndex);
			startIndex= encode(bodyLen, dest, startIndex);
			startIndex= encode(pkt.getSubPkt(), dest, startIndex);
			
			return dest;
		}
		
		public static byte[] getByteArray(PingPacket pkt) {
			int bodyLen= getNumBytes(pkt);
			byte[] dest = new byte[BYTENUM_PKTTYPE + BYTENUM_PKTLENGTH + bodyLen];
			
			int startIndex= 0;
			startIndex= encode(PacketType.PING_PACKET, dest, startIndex);
			startIndex= encode(bodyLen, dest, startIndex);
			
			return dest;
		}
		
		protected static int getSize(SubPkt subPkt){
			return BYTENUM_PKTTYPE + BYTENUM_PKTLENGTH + Marshaller.getNumBytes(subPkt);
		}
		
		protected static int getNumBytes(UnSubPacket pkt) {
			return getSize(pkt.getSubPkt());
		}
		
		protected static int getNumBytes(PingPacket pkt) {
			return 0;
		}

		protected static int encode(PacketType source, byte[] dest, int startIndex) {
			dest[startIndex++]= (byte) source.toValue(); 
			return startIndex;
		}
	}
	
	/**
	 * Flattens the given packet to an array of bytes.
	 */
	public static byte[] marshalRule(Packet pkt, EngineType eType) {
		if (pkt instanceof RulePacket) return MyMarshaller.getByteArray((RulePacket) pkt, eType);
		return null;
	}
	
	/**
	 * Flattens the given packet to an array of bytes.
	 */
	public static byte[] marshal(Packet pkt) {
		if (pkt instanceof PubPacket) return MyMarshaller.getByteArray((PubPacket) pkt);
		else if (pkt instanceof SubPacket) return MyMarshaller.getByteArray((SubPacket) pkt);
		else if (pkt instanceof UnSubPacket) return MyMarshaller.getByteArray((UnSubPacket) pkt);
		else return MyMarshaller.getByteArray((PingPacket) pkt);
	}
}
