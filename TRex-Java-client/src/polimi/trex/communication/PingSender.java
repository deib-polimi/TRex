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

import polimi.trex.packets.PingPkt;

class PingSender implements Runnable {
	private final static long PING_TIMEOUT= 125000;
	
	private TransportManager manager;
	private long lastSendTime;
	private boolean stop;
	
	PingSender(TransportManager manager) {
		this.manager = manager;
		this.lastSendTime = System.currentTimeMillis();
		this.stop = false;
	}
	
	synchronized void updateLastSendTime() {
		lastSendTime = System.currentTimeMillis();
	}
	
	synchronized void stop() {
		stop = true;
	}
	
	@Override
	public void run() {
		PingPkt pkt = new PingPkt();
		while (true) {
			long currentTime = System.currentTimeMillis();
			synchronized (this) {
				if (stop) break;
			}
			long timeToSleep = 0;
			boolean needsToSend = true;
			synchronized (this) {
				if (currentTime-lastSendTime < PING_TIMEOUT/2) {
					needsToSend = false;
				}
				timeToSleep = lastSendTime + PING_TIMEOUT - currentTime;
			}
			if (needsToSend) {
				try {
					manager.send(pkt);
				} catch (IOException e) {
					e.printStackTrace();
				}
				updateLastSendTime();
			}
			if (timeToSleep <= 0) continue;
			try {
				Thread.sleep(timeToSleep);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

}
