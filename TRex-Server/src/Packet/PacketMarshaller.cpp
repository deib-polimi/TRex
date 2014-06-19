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

#include "PacketMarshaller.hpp"

using concept::packet::PacketMarshaller;
using concept::util::CharVector;
using concept::util::CharVectorPtr;

char* PacketMarshaller::encode(PingPkt *pkt){
	char *array = new char[getSize(pkt)];
	int startIndex= 0;

	// encode type
	array[startIndex]= concept::packet::PING_PACKET;
	startIndex += BYTENUM_TYPE;

	// encode length (0)
	Marshaller::encode(0, array, startIndex);

	return array;
}

char* PacketMarshaller::encode(UnSubPkt *pkt){
	char *array = new char[getSize(pkt)];
	int startIndex= 0;

	// encode type
	array[startIndex]= concept::packet::UNSUB_PACKET;
	startIndex += BYTENUM_TYPE;

	// encode length
	int subPktLength= Marshaller::getSize(pkt->getSubPkt());
	Marshaller::encode(subPktLength, array, startIndex);
	startIndex += BYTENUM_LENGTH;

	Marshaller::encode(pkt->getSubPkt(), array, startIndex);

	return array;
}

int PacketMarshaller::getSize(PingPkt *pkt){
	return BYTENUM_TYPE + BYTENUM_LENGTH;
}

int PacketMarshaller::getSize(UnSubPkt *pkt){
	int subPktLength= Marshaller::getSize(pkt->getSubPkt());
	return BYTENUM_TYPE + BYTENUM_LENGTH + subPktLength;
}

CharVectorPtr PacketMarshaller::marshal(PktPtr pkt){
	return boost::apply_visitor(MarshallerVisitor(*this), pkt);
}

CharVectorPtr PacketMarshaller::MarshallerVisitor::operator()(RulePkt * pkt) const{
	Marshaller marshaller= parent;
	char *bytes= marshaller.encode(pkt);
	int size= marshaller.getSize(pkt);
	CharVectorPtr vectorPtr(boost::make_shared<CharVector>(bytes, bytes+size));
	delete[] bytes;
	return vectorPtr;
}

CharVectorPtr PacketMarshaller::MarshallerVisitor::operator()(PubPkt * pkt) const{
	Marshaller marshaller= parent;
	char *bytes= marshaller.encode(pkt);
	int size= marshaller.getSize(pkt);
	CharVectorPtr vectorPtr(boost::make_shared<CharVector>(bytes, bytes+size));
	delete[] bytes;
	return vectorPtr;
}

CharVectorPtr PacketMarshaller::MarshallerVisitor::operator()(SubPkt * pkt) const{
	Marshaller marshaller= parent;
	char *bytes= marshaller.encode(pkt);
	int size= marshaller.getSize(pkt);
	CharVectorPtr vectorPtr(boost::make_shared<CharVector>(bytes, bytes+size));
	delete[] bytes;
	return vectorPtr;
}

CharVectorPtr PacketMarshaller::MarshallerVisitor::operator()(UnSubPkt * pkt) const{
	char *bytes= parent.encode(pkt);
	int size= parent.getSize(pkt);
	CharVectorPtr vectorPtr(boost::make_shared<CharVector>(bytes, bytes+size));
	delete[] bytes;
	return vectorPtr;
}

CharVectorPtr PacketMarshaller::MarshallerVisitor::operator()(PingPkt * pkt) const{
	char *bytes= parent.encode(pkt);
	int size= parent.getSize(pkt);
	CharVectorPtr vectorPtr(boost::make_shared<CharVector>(bytes, bytes+size));
	delete[] bytes;
	return vectorPtr;
}
