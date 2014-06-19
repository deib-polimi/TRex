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
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;

import polimi.trex.common.Consts.EngineType;
import polimi.trex.marshalling.Marshaller;
import polimi.trex.packets.AdvPkt;
import polimi.trex.packets.JoinPkt;
import polimi.trex.packets.PubPkt;
import polimi.trex.packets.RulePkt;
import polimi.trex.packets.SubPkt;
import polimi.trex.packets.TRexPkt;

/**
 * A Communication manages the connection with a T-Rex servers.
 * Sends out packets and forwards received ones to connected PacketListeners.
 */
public class CommunicationManager implements Runnable {
	private int localPort;
	private Socket readSocket = null;
	private Socket writeSocket = null;
	private PacketReader reader = null;
	private volatile boolean isReaderConnected = false;

	/**
	 * Creates a new CommunicationManager that uses a single socket to send and receive data.
	 * Important: this is intended for testing purpose only and should not be used to communicate
	 * with the T-Rex server.
	 * 
	 * @param serverAddress address of the server
	 * @param serverPort port of the server
	 */
	public CommunicationManager(String serverAddress, int serverPort) {
		this.localPort = -1;
		try {
			writeSocket = new Socket(serverAddress, serverPort);
			readSocket = writeSocket;
			reader = new PacketReader();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Creates a new ConnectionManager that uses two sockets to send and receive data.
	 * It connects to the server having the given address and port, and, if needed, listens
	 * on the localPort for event notifications.
	 * 
	 * @param serverAddress address of the server
	 * @param serverPort port of the server
	 * @param localPort local port
	 */
	public CommunicationManager(String serverAddress, int serverPort, int localPort) {
		try {
			writeSocket = new Socket(serverAddress, serverPort);
			reader = new PacketReader();
		} catch (UnknownHostException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		this.localPort = localPort;
		Thread acceptConnectionThread = new Thread(this);
		acceptConnectionThread.setDaemon(true);
		acceptConnectionThread.start();
	}

	public void start() {
		if (readSocket!=null) {
			try {
				reader.startReader(readSocket.getInputStream());
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	public void stop() {
		reader.stopReader();
	}

	public void addPacketListener(PacketListener listener) {
		reader.addPacketListener(listener);
	}

	public void removePacketListener(PacketListener listener) {
		reader.removePacketListener(listener);
	}

	public synchronized void send(TRexPkt pkt, EngineType eType) {
		byte[] byteArray = null;
		if (pkt instanceof PubPkt) {
			PubPkt pubPkt = (PubPkt) pkt;
			byteArray = Marshaller.getByteArray(pubPkt);
		} else if (pkt instanceof RulePkt) {
			RulePkt rulePkt = (RulePkt) pkt;
			byteArray = Marshaller.getByteArray(rulePkt, eType);
		} else if (pkt instanceof SubPkt) {
			if (! isReaderConnected && localPort>0) {
				JoinPkt joinPkt = new JoinPkt(ipToLong("127.0.0.1"), localPort);
				send(joinPkt, eType);
			}
			SubPkt subPkt = (SubPkt) pkt;
			byteArray = Marshaller.getByteArray(subPkt);
		} else if (pkt instanceof AdvPkt) {
			AdvPkt advPkt = (AdvPkt) pkt;
			byteArray = Marshaller.getByteArray(advPkt);
		}	else if (pkt instanceof JoinPkt) {
			JoinPkt joinPkt = (JoinPkt) pkt;
			byteArray = Marshaller.getByteArray(joinPkt);
		} else {
			// Unknown packet: return an exception?
			return;
		}
		try {
			writeSocket.getOutputStream().write(byteArray);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public void run() {
		try {
			ServerSocket serverSocket = new ServerSocket(localPort);
			readSocket = serverSocket.accept();
			reader.startReader(readSocket.getInputStream());
			isReaderConnected = true;
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private long ipToLong(String addr) {
		String[] addrArray = addr.split("\\.");
		long num = 0;
		for (int i=0;i<addrArray.length;i++) {
			int power = 3-i;
			num += ((Integer.parseInt(addrArray[i])%256 * Math.pow(256,power)));
		}
		return num;
	}

}
