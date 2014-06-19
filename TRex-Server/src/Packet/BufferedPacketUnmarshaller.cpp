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

#include "BufferedPacketUnmarshaller.hpp"

using namespace concept::packet;
using concept::util::CharVector;

BufferedPacketUnmarshaller::BufferedPacketUnmarshaller() {
	buffer.reserve(BUFFER_START_LENGTH);
}

PingPkt* BufferedPacketUnmarshaller::decodePingPkt(char *bytes){
	PingPkt *pkt = new PingPkt();
	return pkt;
}

UnSubPkt* BufferedPacketUnmarshaller::decodeUnSubPkt(char *bytes){
	// skip SubPkt's type and length
	int startIndex= PacketMarshaller::BYTENUM_TYPE + PacketMarshaller::BYTENUM_LENGTH;
	SubPkt* subPkt= decodeSubPkt(bytes, startIndex);
	return new UnSubPkt(*subPkt);
}

concept::packet::PktType BufferedPacketUnmarshaller::decodePktType(char byte){
	concept::packet::PktType type= static_cast<concept::packet::PktType>(byte);
	switch (type) {
		case RULE_PACKET: return RULE_PACKET;
		case PUB_PACKET: return PUB_PACKET;
		case SUB_PACKET: return SUB_PACKET;
		case PING_PACKET: return PING_PACKET;
		case UNSUB_PACKET: return UNSUB_PACKET;
	}
	throw std::logic_error("Packet type not existent or not handled.");
}

std::size_t BufferedPacketUnmarshaller::decodePktLength(char* bytes){
	return decodeInt(bytes);
}

PktPtr* BufferedPacketUnmarshaller::unmarshalOne(CharVector & bytes){
	// Number of bytes already processed
	std::size_t processed= 0;

	// Checks to have received enough bytes to decode the packet type, otherwise return empty pointer
	if (bytes.size()-processed < PacketMarshaller::BYTENUM_TYPE) return NULL;

	// Decodes the packet type
	concept::packet::PktType pktType = decodePktType(bytes[0]);
	processed += PacketMarshaller::BYTENUM_TYPE;

	// Checks to have received enough bytes to decode the packet length, otherwise return empty pointer
	if (bytes.size()-processed < PacketMarshaller::BYTENUM_LENGTH) return NULL;

	// Decodes the packet length
	// &bytes[0] is valid since a std::vector stores memory contiguously
	std::size_t pktLength = decodePktLength(&bytes[0]+processed);
	processed += PacketMarshaller::BYTENUM_LENGTH;

	// Checks to have received enough bytes to decode the packet body, otherwise return empty pointer
	if (bytes.size()-processed < pktLength) return NULL;

	// Decodes the packet body and process it
	PktPtr *pkt= new PktPtr();
	switch (pktType) {
		case PUB_PACKET:
			*pkt= decodePubPkt(&bytes[0]+processed);
			break;
		case RULE_PACKET:
			*pkt= decodeRulePkt(&bytes[0]+processed);
			break;
		case SUB_PACKET:
			*pkt= decodeSubPkt(&bytes[0]+processed);
			break;
		case UNSUB_PACKET:
			*pkt= decodeUnSubPkt(&bytes[0]+processed);
			break;
		case PING_PACKET:
			*pkt= decodePingPkt(&bytes[0]+processed);
			break;
	}

	processed += pktLength;
	// Deletes all processed bytes; possibly some bytes of next packet
	// that were already received are kept, to be processed next time
	bytes.erase(bytes.begin(), bytes.begin()+processed);

	return pkt;
}

std::vector<PktPtr> BufferedPacketUnmarshaller::unmarshalAll(){
	std::vector<PktPtr> pkts;
	// Unmarshal as many packets as possible
	PktPtr* pkt;
	while ((pkt= unmarshalOne(buffer)) != NULL){
		pkts.push_back(*pkt);
		delete pkt;
	}

	return pkts;
}
