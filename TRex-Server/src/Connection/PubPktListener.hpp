/*
 * PubPktListener.hpp
 *
 *  Created on: Dec 12, 2012
 *      Author: margara
 */

#ifndef PUBPKTLISTENER_HPP_
#define PUBPKTLISTENER_HPP_

#include "../external.hpp"
#include "../util.hpp"

namespace concept{
namespace connection{

class PubPktListener{
	public:
		virtual ~PubPktListener();

		virtual void processPubPkt(PubPkt *pkt) = 0;
};

} // connection
} // concept



#endif /* PUBPKTLISTENER_HPP_ */
