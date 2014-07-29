//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Daniele Rogora
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

package polimi.trex.common;

import java.util.ArrayList;

import polimi.trex.packets.PubPkt;
import polimi.trex.packets.SubPkt;

public class SubscriptionsTable {
	private ArrayList<SubPkt> subscriptions;
	private ArrayList<SubPkt> subscriptionsWithCustomMatcher;

	public SubscriptionsTable() {
		this.subscriptions = new ArrayList<SubPkt>();
		this.subscriptionsWithCustomMatcher = new ArrayList<SubPkt>();
	}
	
	public void addSubscription(SubPkt sub) {
		this.subscriptions.add(sub);
		if (sub.hasCustomMatcher()) this.subscriptionsWithCustomMatcher.add(sub);
	}
	
	public void removeSubscription(SubPkt sub) {
		this.subscriptions.remove(sub);
		if (sub.hasCustomMatcher()) this.subscriptionsWithCustomMatcher.remove(sub);
	}
	
	public ArrayList<SubPkt> getSubscriptions() {
		return this.subscriptions;
	}
	
	public boolean match(PubPkt pkt) {
		if (this.subscriptionsWithCustomMatcher.size()==0) {
			//In this case all the matching work has already been done on the server
			return true;
		}
		//Else we have to check again the match already done on the server, plus our custom matcher
		for (SubPkt sub: this.subscriptionsWithCustomMatcher) {
			//When we find one matching we don't care about the others; it's enough to send the notification
			//1 means that constraint matching succeded but the custom matcher failed
			if (sub.match(pkt)==-1) return false;
		}
		//Ok if I'm here it means that (1 must hold):
		// - I have some subs with custom matcher but none of them have matched against the static constraints 
		// - I have some subs with custom matcher but none of them have not matched the custom matcher while matching the constraints
		return true;
	}
}
