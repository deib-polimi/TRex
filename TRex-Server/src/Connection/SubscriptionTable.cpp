#include "SubscriptionTable.hpp"

using namespace std;
using concept::connection::SubscriptionTable;
using concept::connection::PubPktListener;

SubscriptionTable::SubscriptionTable() { }

SubscriptionTable::~SubscriptionTable() { }

void SubscriptionTable::addListener(PubPktListener *listener) {
	listeners.insert(listener);
}

void SubscriptionTable::removeListener(PubPktListener *listener) {
	listeners.erase(listener);
}

void SubscriptionTable::processPublication(PubPkt *pkt) {
	for (set<PubPktListener *>::iterator it=listeners.begin(); it!=listeners.end(); ++it) {
		PubPktListener *list = *it;
		list->processPubPkt(pkt);
	}
}
