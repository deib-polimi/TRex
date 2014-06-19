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

package polimi.trex.communication;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;

import polimi.trex.marshalling.Unmarshaller;
import polimi.trex.packets.TRexPkt;
import polimi.trex.packets.TRexPkt.PktType;


public class PacketReader implements Runnable {

	private InputStream inputStream;
	private Collection<PacketListener> packetListeners;
	private volatile boolean running;
	private volatile boolean stop;
	private Thread t;
	
	public PacketReader() {
		this.packetListeners = new ArrayList<PacketListener>();
		this.running = false;
		this.stop = false;
	}
	
	public synchronized void startReader(InputStream inputStream) {
		stop = false;
		if (! running) {
			this.inputStream = inputStream;
			t = new Thread(this);
			t.start();
		}
	}
	
	public synchronized void stopReader() {
		stop = true;
	}
	
	public synchronized void addPacketListener(PacketListener packetListener) {
		packetListeners.add(packetListener);
	}
	
	public synchronized void removePacketListener(PacketListener packetListener) {
		packetListeners.remove(packetListener);
	}
	
	@Override
	public void run() {
		while(true) {
			// Reads the packet type
			byte[] typeByteArray = new byte[1];
			try {
				inputStream.read(typeByteArray, 0, 1);
			} catch (IOException e) {
				e.printStackTrace();
			}
			PktType type = Unmarshaller.decodePktType(typeByteArray);
			// Reads the packet length
			byte[] lengthByteArray = new byte[4];
			int alreadyRead = 0;
			while (alreadyRead < 4) {
				try {
					alreadyRead += inputStream.read(lengthByteArray, alreadyRead, 4-alreadyRead);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			int length = Unmarshaller.decodeInt(lengthByteArray);
			// Reads the packet
			byte[] pktByteArray = new byte[length];
			alreadyRead = 0;
			while (alreadyRead < length) {
				try {
					alreadyRead += inputStream.read(pktByteArray, alreadyRead, length-alreadyRead);
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			TRexPkt pkt = null;
			if (type == PktType.PUB_PKT) pkt = Unmarshaller.decodePubPkt(pktByteArray);
			else if (type == PktType.RULE_PKT) pkt = Unmarshaller.decodeRulePkt(pktByteArray);
			else if (type == PktType.SUB_PKT) pkt = Unmarshaller.decodeSubPkt(pktByteArray);
			else if (type == PktType.ADV_PKT) pkt = Unmarshaller.decodeAdvPkt(pktByteArray);
			else if (type == PktType.JOIN_PKT) pkt = Unmarshaller.decodeJoinPkt(pktByteArray);
			// Delivers received packet to all connected listeners
			synchronized (this) {
				for (PacketListener packetListener : packetListeners) {
					packetListener.notifyPktReceived(pkt);
				}
			}
			// Checks whether it should stop
			synchronized (this) {
				if (stop) {
					running = false;
					break;
				}
			}
		}
	}
}
