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

#ifndef REQUESTHANDLER_H_
#define REQUESTHANDLER_H_

#include "../external.hpp"
#include "../packet.hpp"
#include "../util.hpp"

#include "SubscriptionTable.hpp"
#include "ConnectionProxy.hpp"
#include "PubPktListener.hpp"

namespace concept{
namespace connection{

class RequestHandler : public PubPktListener {
public:
#ifdef HAVE_GTREX
	RequestHandler(TRexEngine &tRexEngine, GPUEngine &gtRexEngine, ConnectionProxy& connection, SubscriptionTable &subTable, bool useGPUPar);
#else
	RequestHandler(TRexEngine &tRexEngine, ConnectionProxy& connection, SubscriptionTable &subTable, bool useGPUPar);
#endif
	~RequestHandler();

	template <std::size_t N>  // defined here because of the template
	void handleRequest(concept::util::CharArray<N> & recBytes, std::size_t recBytesNum){
		std::vector<concept::packet::PktPtr> pkts= unmarshaller.unmarshal<N>(recBytes, recBytesNum);
		handleRequest(pkts);
	}

private:

	class PktHandleVisitor : public boost::static_visitor<>{
		RequestHandler &parent;
	public:
		PktHandleVisitor(RequestHandler& p, bool useGPUPar): parent(p), useGPU(useGPUPar) { }
		void operator()(RulePkt * pkt) const;
		void operator()(PubPkt * pkt) const;
		void operator()(SubPkt * pkt) const;
		void operator()(concept::packet::UnSubPkt * pkt) const;
		void operator()(concept::packet::PingPkt * pkt) const;
		
	private:
	    	bool useGPU;
	};

	class ResultListenerImpl: public ResultListener{
		RequestHandler &parent;
	public:
		ResultListenerImpl(RequestHandler& p): parent(p) { }
		virtual ~ResultListenerImpl() { }
		void handleResult(std::set<PubPkt *> &genPkts, double procTime);
	};
	
	TRexEngine &tRexEngine;
	
#ifdef HAVE_GTREX
	GPUEngine &gtRexEngine;
#endif
	ConnectionProxy& connection;
	ResultListenerImpl resultListener;
	SubscriptionTable &subTable;
  	bool useGPU;


	concept::packet::BufferedPacketUnmarshaller unmarshaller;
	concept::packet::PacketMarshaller marshaller;

	std::vector<SubPkt> subscriptions;
	bool firstSubscriptionDone;

	concept::util::CharVector lastRequest;

	void handleRequest(std::vector<concept::packet::PktPtr> & pkts);
	void handleRulePkt(RulePkt* pkt);
	void handlePubPkt(PubPkt* pkt);
	void handleSubPkt(SubPkt* pkt);
	void handleAdvPkt(AdvPkt* pkt);

	void processPubPkt(PubPkt *pkt);
};

} // connection
} // concept

#endif /* REQUESTHANDLER_H_ */
