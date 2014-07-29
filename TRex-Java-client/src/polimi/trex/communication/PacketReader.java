//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Francesco Feltrinelli
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
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.Collection;

import polimi.trex.common.SubscriptionsTable;
import polimi.trex.marshalling.BufferedPacketUnmarshaller;
import polimi.trex.packets.PubPkt;
import polimi.trex.packets.TRexPkt;

public class PacketReader implements Runnable {

	private InputStream inputStream;
	private Collection<PacketListener> listeners;
	private volatile boolean running;
	private volatile boolean stop;
	private Thread t;
	
	private final static int BUFFER_LENGTH= 1024;
	private byte[] buffer;
	private BufferedPacketUnmarshaller unmarshaller;
	
	private SubscriptionsTable sTable;
	
	public PacketReader(SubscriptionsTable sT) {
		this.listeners = new ArrayList<PacketListener>();
		this.running = false;
		this.stop = false;
		this.sTable = sT;
		buffer= new byte[BUFFER_LENGTH];
		unmarshaller= new BufferedPacketUnmarshaller();
	}
	
	public PacketReader(InputStream inputStream, SubscriptionsTable sT){
		this(sT);
		this.inputStream= inputStream;
	}
	
	public synchronized void startReader() {
		stop = false;
		if (! running) {
			t = new Thread(this);
			t.start();
		}
	}
	
	public synchronized void stopReader() {
		stop = true;
	}
	
	public void addPacketListener(PacketListener packetListener) {
		synchronized(listeners) {
			listeners.add(packetListener);
		}
	}
	
	public void removePacketListener(PacketListener packetListener) {
		synchronized(listeners) {
			listeners.remove(packetListener);
		}
	}
	
	public void setInputStream(InputStream inputStream) {
		this.inputStream = inputStream;
	}

	@Override
	public void run() {
		
		try {
			while (true) {
				try {
					// Read bytes from input stream
					int numRead= inputStream.read(buffer);
					if (numRead < 0) throw new IOException("End of stream reached");
					
					TRexPkt[] pkts= unmarshaller.unmarshal(buffer, 0, numRead);
					boolean matched = true;
					synchronized (listeners) {
						for (TRexPkt pkt: pkts){
							//if the packet is a PubPkt and I have any custom matcher I need to redo all the post filtering process
							if (this.sTable != null && pkt instanceof PubPkt) {
								matched = false;
								if (this.sTable.match((PubPkt) pkt)) {
									matched = true;
								}
							}
							if (!matched) continue;
							// Deliver received packet to all connected listeners
							for (PacketListener listener : listeners) {
								listener.notifyPktReceived(pkt);
							}
						}
						
					}
				} catch (InterruptedIOException e) {
					// timeout for blocking receive expired: do nothing
				}
				// Check whether it should stop
				synchronized (this) {
					if (stop) {
						running = false;
						break;
					}
				}
			}
		} catch (IOException e) {
			// error with connection: signal error and exit
			e.printStackTrace();
			for (PacketListener listener : listeners) {
				listener.notifyConnectionError();
			}
		}
	}
}
