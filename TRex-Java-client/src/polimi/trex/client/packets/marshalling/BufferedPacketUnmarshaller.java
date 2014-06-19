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

import java.util.ArrayList;

import polimi.trex.client.packets.Packet;
import polimi.trex.client.utils.CollectionUtils;
import polimi.trex.client.utils.MutableInt;



/**
 * A buffered {@link Packet}'s unmarshaller. This is suitable where the total number
 * of bytes is not known a priori and bytes are given to this decoder little by little,
 * as in network transfers. If you already have all the bytes to be decoded, use
 * {@link PacketUnmarshaller}, which is more efficient.
 * 
 * This uses {@link PacketUnmarshaller}, which is compatible with the encoding
 * used in {@link PacketMarshaller}. As {@link PacketMarshaller}, this supposes
 * that given bytes are logically valid (there are no byte errors, they belong to a 
 * proper continous stream of {@link Packet}s, and so on): no validity checks are
 * done apart from {@link Packet}'s length and the result would be unpredictable.
 * 
 * @author Francesco Feltrinelli
 *
 */
public class BufferedPacketUnmarshaller {
	
	private byte[] buffer;
	
	public BufferedPacketUnmarshaller(){
		buffer= new byte[0];
	}
	
	/**
	 * Copies the given bytes to internal buffer and tries to decode from all the bytes
	 * in the buffer as many {@link Packet}s as possible. Remaining bytes are left
	 * in the buffer for next time this is called. 
	 * If the bytes are not enough to decode a packet an empty array is returned.
	 * The given byte array is not changed.
	 * 
	 * @param pktBytes the array of bytes from which bytes will be copied
	 * @param start the starting offset in the array of bytes
	 * @param length the number of bytes to be copied from the array
	 * @return the (possibly empty) array of decoded {@link Packet}s
	 * @throws ArrayIndexOutOfBoundsException if an illegal start or length
	 * is specified
	 */
	public Packet[] unmarshal(byte[] pktBytes, int start, int length){
		if (start<0 || start+length>pktBytes.length) 
			throw new ArrayIndexOutOfBoundsException();
		
		buffer= CollectionUtils.concat(buffer, pktBytes, start, length);
		ArrayList<Packet> pkts= new ArrayList<Packet>();
		
		Packet pkt= null;
		MutableInt offset= new MutableInt(0);
		do {
			pkt= PacketUnmarshaller.unmarshal(buffer, offset);
			if (pkt != null){
				pkts.add(pkt);
				// Remove decoded bytes from buffer
				buffer= CollectionUtils.subset(buffer, offset.get(), buffer.length-offset.get());
				offset.setValue(0);
			}
		} while (pkt != null);
		
		return pkts.toArray(new Packet[pkts.size()]);
	}
	
	/**
	 * Equivalent of {@link #unmarshal(byte[], int, int)} where all the bytes of
	 * the given array are taken for decoding.
	 */
	public Packet[] unmarshal(byte[] pktBytes){
		return unmarshal(pktBytes, 0, pktBytes.length);
	}
	
	/**
	 * Clear internal buffer.
	 */
	public void clear(){
		buffer= new byte[0];
	}
}