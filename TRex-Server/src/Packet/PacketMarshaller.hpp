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

#ifndef PACKETMARSHALLER_H_
#define PACKETMARSHALLER_H_

#include "../util.hpp"

#include "Pkt.hpp"

namespace concept{
namespace packet{

class PacketMarshaller : public Marshaller{
public:
	PacketMarshaller() { }
	virtual ~PacketMarshaller() { }

	concept::util::CharVectorPtr marshal(concept::packet::PktPtr pkt);

	// Number of bytes used to encode the packet type
	static const std::size_t BYTENUM_TYPE= 1;
	// Number of bytes used to encode the packet length
	static const std::size_t BYTENUM_LENGTH= 4;

private:

	class MarshallerVisitor : public boost::static_visitor<concept::util::CharVectorPtr>{
		PacketMarshaller &parent;
	public:
		MarshallerVisitor(PacketMarshaller& parent): parent(parent) { }

		concept::util::CharVectorPtr operator()(RulePkt * pkt) const;
		concept::util::CharVectorPtr operator()(PubPkt * pkt) const;
		concept::util::CharVectorPtr operator()(SubPkt * pkt) const;
		concept::util::CharVectorPtr operator()(concept::packet::UnSubPkt * pkt) const;
		concept::util::CharVectorPtr operator()(concept::packet::PingPkt * pkt) const;
	};

	char* encode(concept::packet::PingPkt *pkt);
	char* encode(concept::packet::UnSubPkt *pkt);

	int getSize(concept::packet::PingPkt *pkt);
	int getSize(concept::packet::UnSubPkt *pkt);
};

} // packet
} // concept

#endif /* PACKETMARSHALLER_H_ */
