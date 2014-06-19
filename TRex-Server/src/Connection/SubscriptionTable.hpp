#ifndef SUBSCRIPTIONTABLE_HPP_
#define SUBSCRIPTIONTABLE_HPP_

#include <set>
#include "../packet.hpp"
#include "PubPktListener.hpp"

namespace concept{
namespace connection{

class SubscriptionTable {
public:
	SubscriptionTable();

	virtual ~SubscriptionTable();

	void addListener(PubPktListener *listener);

	void removeListener(PubPktListener *listener);

	void processPublication(PubPkt *pkt);

private:

	std::set<PubPktListener *> listeners;
};

}
}

#endif /* SUBSCRIPTIONTABLE_HPP_ */
