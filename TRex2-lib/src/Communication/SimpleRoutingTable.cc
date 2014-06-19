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

#include "SimpleRoutingTable.h"

using namespace std;

SimpleRoutingTable::SimpleRoutingTable() {
	mutex = new pthread_mutex_t;
	pthread_mutex_init(mutex, NULL);
}

SimpleRoutingTable::~SimpleRoutingTable() {
	pthread_mutex_destroy(mutex);
	delete mutex;
	for (map<int, map<int, list<SubPkt *> > >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
		for (map<int, list<SubPkt *> >::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
			for (list<SubPkt *>::iterator it3=it2->second.begin(); it3!=it2->second.end(); ++it3) {
				SubPkt *pkt = *it3;
				delete pkt;
			}
		}
	}
}

void SimpleRoutingTable::installSubscription(int clientId, SubPkt *subscription) {
	pthread_mutex_lock(mutex);
	int type = subscription->getEventType();
	map<int, map<int, list<SubPkt *> > >::iterator it = subscriptions.find(type);
	if (it==subscriptions.end()) {
		list<SubPkt *> newList;
		newList.push_back(subscription);
		subscriptions[type].insert(make_pair(clientId, newList));
	} else {
		map<int, list<SubPkt *> >::iterator it2 = it->second.find(clientId);
		if (it2==it->second.end()) {
			list<SubPkt *> newSet;
			newSet.push_back(subscription);
			it->second.insert(make_pair(clientId, newSet));
		} else {
			it2->second.push_back(subscription);
		}
	}
	pthread_mutex_unlock(mutex);
}

void SimpleRoutingTable::deleteSubscription(int clientId, SubPkt *subscription) {
	pthread_mutex_lock(mutex);
	int type = subscription->getEventType();
	map<int, map<int, list<SubPkt *> > >::iterator it=subscriptions.find(type);
	if (it==subscriptions.end()) {
		pthread_mutex_unlock(mutex);
		return;
	}
	map<int, list<SubPkt *> >::iterator it2 = it->second.find(clientId);
	if (it2==it->second.end()) {
		pthread_mutex_unlock(mutex);
		return;
	}
	for (list<SubPkt *>::iterator it3=it2->second.begin(); it3!=it2->second.end(); ) {
		SubPkt *stored = *it3;
		if (sameSubscription(stored, subscription)) it3=it2->second.erase(it3);
		else ++it3;
	}
	pthread_mutex_unlock(mutex);
}

void SimpleRoutingTable::removeClient(int clientId) {
	pthread_mutex_lock(mutex);
	for (map<int, map<int, list<SubPkt *> > >::iterator it=subscriptions.begin(); it!=subscriptions.end(); ++it) {
		if (it->second.find(clientId)==it->second.end()) continue;
		for (list<SubPkt *>::iterator it2=it->second[clientId].begin(); it2!=it->second[clientId].end(); ++it2) {
			SubPkt *pkt = *it2;
			delete pkt;
		}
		it->second.erase(clientId);
	}
	pthread_mutex_unlock(mutex);
}

void SimpleRoutingTable::getMatchingClients(PubPkt *pubPkt, set<int> &clients) {
	pthread_mutex_lock(mutex);
	int eventType = pubPkt->getEventType();
	map<int, map<int, list<SubPkt *> > >::iterator it=subscriptions.find(eventType);
	if (it==subscriptions.end()) {
		pthread_mutex_unlock(mutex);
		return;
	}
	for (map<int, list<SubPkt *> >::iterator it2=it->second.begin(); it2!=it->second.end(); ++it2) {
		int client = it2->first;
		for (list<SubPkt *>::iterator it3=it2->second.begin(); it3!=it2->second.end(); ++it3) {
			SubPkt *subPkt = *it3;
			if (matches(pubPkt, subPkt)) {
				clients.insert(client);
				break;
			}
		}
	}
	pthread_mutex_unlock(mutex);
}

bool SimpleRoutingTable::sameSubscription(SubPkt *s1, SubPkt *s2) {
	if (s1->getEventType()!=s2->getEventType()) return false;
	int numConstraints = s1->getConstraintsNum();
	if (numConstraints!=s2->getConstraintsNum()) return false;
	for (int i=0; i<numConstraints; i++) {
		if (! (s1->getConstraint(i)==s2->getConstraint(i))) return false;
	}
	return true;
}

bool SimpleRoutingTable::matches(PubPkt *pubPkt, SubPkt *subPkt) {
	for (int c=0; c<subPkt->getConstraintsNum(); c++) {
		Constraint constr = subPkt->getConstraint(c);
		int index;
		ValType valType;
		if (! pubPkt->getAttributeIndexAndType(constr.name, index, valType)) return false;
		if (valType!=constr.type) continue;
		if (valType==INT) {
			if (constr.op==EQ && pubPkt->getIntAttributeVal(index)!=constr.intVal) return false;
			if (constr.op==LT && pubPkt->getIntAttributeVal(index)>=constr.intVal) return false;
			if (constr.op==GT && pubPkt->getIntAttributeVal(index)<=constr.intVal) return false;
			if (constr.op==DF && pubPkt->getIntAttributeVal(index)==constr.intVal) return false;
		} else if (constr.type==FLOAT) {
			if (constr.op==EQ && pubPkt->getFloatAttributeVal(index)!=constr.floatVal) return false;
			if (constr.op==LT && pubPkt->getFloatAttributeVal(index)>=constr.floatVal) return false;
			if (constr.op==GT && pubPkt->getFloatAttributeVal(index)<=constr.floatVal) return false;
			if (constr.op==DF && pubPkt->getFloatAttributeVal(index)==constr.floatVal) return false;
		} else if (constr.type==BOOL) {
			if (constr.op==EQ && pubPkt->getBoolAttributeVal(index)!=constr.boolVal) return false;
			if (constr.op==DF && pubPkt->getBoolAttributeVal(index)==constr.boolVal) return false;
		} else if (constr.type==STRING) {
			char stringVal[STRING_VAL_LEN];
			pubPkt->getStringAttributeVal(index, stringVal);
			if (constr.op==EQ && strcmp(stringVal, constr.stringVal)!=0) return false;
			if (constr.op==IN) {
				string s = stringVal;
				if (s.find(constr.stringVal)==string::npos) return false;
			}
		}
	}
	return true;
}
