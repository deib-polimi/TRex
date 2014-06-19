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
import polimi.trex.client.utils.MutableInt;
import polimi.trex.marshalling.Unmarshaller;
import polimi.trex.packets.PubPkt;
import polimi.trex.packets.RulePkt;
import polimi.trex.packets.SubPkt;

/**
 * A {@link Packet}'s unmarshaller. Tries to restore from the given byte array
 * the corresponding {@link Packet}. No particular checks are done on the bytes
 * to ensure they represent a valid {@link Packet}, apart from its length:
 * if invalid bytes are given the result is unpredictable. If bytes are transmitted
 * on network, you should use a "reliable" protocol as TCP.
 * 
 * This is compatible with the encoding used in {@link PacketMarshaller}.
 * 
 * @author Francesco Feltrinelli
 *
 */
public class PacketUnmarshaller {

	private static class MyUnmarshaller extends Unmarshaller {	
		public static PubPacket decodePubPkt(byte[] source, int startIndex) {
			IndexWrapper index = new IndexWrapper();
			index.inc(startIndex);
			PubPkt trexPubPkt= decodePubPkt(source, index);
			return new PubPacket(trexPubPkt);
		}
		
		public static RulePacket decodeRulePkt(byte[] source, int startIndex) {
			IndexWrapper index = new IndexWrapper();
			index.inc(startIndex);
			RulePkt trexRulePkt= decodeRulePkt(source, index);
			return new RulePacket(trexRulePkt);
		}
		
		public static SubPacket decodeSubPkt(byte[] source, int startIndex) {
			IndexWrapper index = new IndexWrapper();
			index.inc(startIndex);
			SubPkt trexSubPkt= decodeSubPkt(source, index);
			return new SubPacket(trexSubPkt);
		}
		
		public static UnSubPacket decodeUnSubPkt(byte[] source, int startIndex) {
			// skip SubPkt's type and length
			startIndex += PacketMarshaller.BYTENUM_PKTTYPE + PacketMarshaller.BYTENUM_PKTLENGTH;
			SubPkt subPkt= decodeSubPkt(source, startIndex);
			return new UnSubPacket(subPkt);
		}
		
		public static PingPacket decodePingPkt(byte[] source, int startIndex){
			return new PingPacket();
		}
		
		public static PacketType decodePktType(byte[] source, int startIndex){
			return PacketType.fromValue(source[startIndex]);
		}
		
		public static int decodePktLength(byte[] source, int startIndex){
			IndexWrapper index= new IndexWrapper();
			index.inc(startIndex);
			return decodeInt(source, index);
		}
	}
	
	/**
	 * Tries to unmarshal a {@link Packet} from the given (supposed valid) byte array.
	 * If the bytes are not enough to decode the packet completely, {@code null}
	 * is returned.
	 * 
	 * @param buffer the bytes' buffer
	 * @param offset the index on the given buffer where to start the decoding from;
	 * this is a {@link MutableInt} so that the caller at the end can know how many 
	 * bytes of the buffer were actually read
	 * @return the decoded packet or {@code null} if the bytes were not enough to
	 * decode it completely
	 */
	public static Packet unmarshal(byte[] buffer, MutableInt offset){
		// Try to decode packet type
		if (buffer.length-offset.get() < PacketMarshaller.BYTENUM_PKTTYPE) return null;
		PacketType type = MyUnmarshaller.decodePktType(buffer, offset.get());
		offset.add(PacketMarshaller.BYTENUM_PKTTYPE);
		
		// Try to decode packet length
		if (buffer.length-offset.get() < PacketMarshaller.BYTENUM_PKTLENGTH) return null;
		int length = MyUnmarshaller.decodePktLength(buffer, offset.get());
		offset.add(PacketMarshaller.BYTENUM_PKTLENGTH);
		
		// Try to decode packet body
		if (buffer.length-offset.get() < length) return null;
		Packet pkt = null;
		switch (type) {
		case PUB_PACKET: 
			pkt= MyUnmarshaller.decodePubPkt(buffer, offset.get());
			break;
		case RULE_PACKET:
			pkt= MyUnmarshaller.decodeRulePkt(buffer, offset.get());
			break;
		case SUB_PACKET: 
			pkt= MyUnmarshaller.decodeSubPkt(buffer, offset.get());
			break;
		case UNSUB_PACKET: 
			pkt= MyUnmarshaller.decodeUnSubPkt(buffer, offset.get());
			break;
		case PING_PACKET: 
			pkt= MyUnmarshaller.decodePingPkt(buffer, offset.get());
			break;
		}
		offset.add(length);
		
		return pkt;
	}
	
	/**
	 * As {@link #unmarshal(byte[], MutableInt)} with 0 as start index.
	 */
	public static Packet unmarshal(byte[] buffer){
		return unmarshal(buffer, new MutableInt(0));
	}
}
