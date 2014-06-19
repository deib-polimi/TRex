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

package polimi.trex.client.communication;

import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.util.ArrayList;
import java.util.Collection;

import polimi.trex.client.packets.Packet;
import polimi.trex.client.packets.marshalling.BufferedPacketUnmarshaller;

public class PacketReader implements Runnable {

	private InputStream inputStream;
	private Collection<PacketListener> listeners;
	private volatile boolean running;
	private volatile boolean stop;
	private Thread t;
	
	private final static int BUFFER_LENGTH= 1024;
	private byte[] buffer;
	private BufferedPacketUnmarshaller unmarshaller;
	
	public PacketReader() {
		this.listeners = new ArrayList<PacketListener>();
		this.running = false;
		this.stop = false;
		
		buffer= new byte[BUFFER_LENGTH];
		unmarshaller= new BufferedPacketUnmarshaller();
	}
	
	public PacketReader(InputStream inputStream){
		this();
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
					
					Packet[] pkts= unmarshaller.unmarshal(buffer, 0, numRead);
					
					// Deliver received packet to all connected listeners
					synchronized (listeners) {
						for (PacketListener listener : listeners) {
							for (Packet pkt: pkts){
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
