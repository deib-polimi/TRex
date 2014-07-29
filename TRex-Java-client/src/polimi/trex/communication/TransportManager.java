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
import java.io.OutputStream;
import java.net.Socket;

import polimi.trex.common.SubscriptionsTable;
import polimi.trex.common.Consts.EngineType;
import polimi.trex.marshalling.Marshaller;
import polimi.trex.packets.RulePkt;
import polimi.trex.packets.SubPkt;
import polimi.trex.packets.TRexPkt;
import polimi.trex.packets.UnSubPkt;

/**
 * A TransportManager manages the connection with a T-Rex server.
 * Sends out packets and forwards received ones to connected PacketListeners.
 */
public class TransportManager {

	private static final int RECEIVE_TIMEOUT= 2000;

	private Socket sock;
	private OutputStream out;
	private PacketReader reader;
	private PingSender pingSender;
	private Thread pingSenderThread;
	private boolean usePing;
	private boolean connected;
	private boolean started;
	private SubscriptionsTable sTable;
    
	/**
	 * Initializes the TransportManager.
	 * If usePing is set to true, also send periodic ping packets to the server
	 * 
	 * @param usePing if true, the TransportManager sends periodic ping packets to the server
	 */
	public TransportManager(boolean usePing) {
		this.sTable = new SubscriptionsTable();
		this.reader = new PacketReader(this.sTable);
		this.usePing = usePing;
		this.connected = false;
		if (usePing) {
			pingSender = new PingSender(this);
			pingSenderThread = new Thread(pingSender);
		}
	}

	/**
	 * Connects with the server having the given address and port
	 * 
	 * @param address The address of the server
	 * @param port the port of the server
	 * @throws IOException
	 */
	public synchronized void connect(String address, int port) throws IOException {
		if (connected) return;
		sock = new Socket(address, port);
		sock.setSoTimeout(RECEIVE_TIMEOUT);
		out= sock.getOutputStream();
		reader.setInputStream(sock.getInputStream());
		connected = true;
	}

	/**
	 * Starts listening for new packets from the server
	 */
	public synchronized void start() {
		if (started) return;
		reader.startReader();
		if (usePing) pingSenderThread.start();
		started = true;
	}

	/**
	 * Stops listening for new packets from the server
	 */
	public synchronized void stop() {
		if (usePing) pingSender.stop();
		reader.stopReader();
		closeSocket();
		started = false;
		connected = false;
	}

	/**
	 * Registers a listener for receiving packets from the server
	 * 
	 * @param listener The listener to register
	 */
	public void addPacketListener(PacketListener listener) {
		reader.addPacketListener(listener);
	}

	/**
	 * Removes one listener of packets
	 * 
	 * @param listener The listener to remove
	 */
	public void removePacketListener(PacketListener listener) {
		reader.removePacketListener(listener);
	}

	
	public void sendRule(TRexPkt pkt, EngineType eType) throws IOException {
		if (out != null) {
			byte[] bytes;
			bytes = Marshaller.marshalRule(pkt, eType);
			out.write(bytes);
			// TODO: for future usage (when the server recognizes every packet as valid ping)
			// if (usePing) pingSender.updateLastSendTime();
		}
	}
	/**
	 * Sends a packet to the server
	 * 
	 * @param pkt The packet to send
	 * @throws IOException
	 */
	public void send(TRexPkt pkt) throws IOException {
		if (out != null) {
			byte[] bytes;
			bytes = Marshaller.marshal(pkt);
			out.write(bytes);
			// TODO: for future usage (when the server recognizes every packet as valid ping)
			// if (usePing) pingSender.updateLastSendTime();
			if (pkt instanceof SubPkt) this.sTable.addSubscription((SubPkt) pkt);
			if (pkt instanceof UnSubPkt) this.sTable.removeSubscription(((UnSubPkt)pkt).getSubPkt());
		}
	}

	/**
	 * There is a separate method other than {@link #send(Packet)}
	 * just for {@link RulePkt} as {@link RulePkt}s are not
	 * TRexClient {@link Packet}s. This is only for testing.
	 */
	/*
	public void send(RulePkt pkt) throws IOException {
		if (out != null) {
			byte[] bytes = Marshaller.getByteArray(pkt);
			out.write(bytes);
			// TODO: for future usage (when the server recognizes every packet as valid ping)
			// if (usePing) pingSender.updateLastSendTime();
		}
	}
	*/

	private void closeSocket(){
		try {
			if (sock != null && !sock.isClosed()) {
				sock.close();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * The library user may be interested in the subTable to make unsubscriptions
	 * @return reference to the subscription table
	 */
	public SubscriptionsTable getSubscriptionsTable() {
		return this.sTable;
	}
}
