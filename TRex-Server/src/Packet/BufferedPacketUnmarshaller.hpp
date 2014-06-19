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

#ifndef BUFFEREDPACKETUNMARSHALLER_H_
#define BUFFEREDPACKETUNMARSHALLER_H_

#include "../util.hpp"

#include "PacketMarshaller.hpp"

namespace concept{
namespace packet{

class BufferedPacketUnmarshaller : public Unmarshaller{
public:
	BufferedPacketUnmarshaller();
	virtual ~BufferedPacketUnmarshaller() { };

	template <int N> // defined here because of the template
	std::vector<concept::packet::PktPtr> unmarshal(concept::util::CharArray<N> & bytes, std::size_t length){
		// Append bytes to buffer
		buffer.insert(buffer.end(), bytes.begin(), bytes.begin()+length);
		return unmarshalAll();
	}

private:
	// Initial space reserved for buffer
	static const std::size_t BUFFER_START_LENGTH= 2048;


	concept::util::CharVector buffer;

	concept::packet::PingPkt* decodePingPkt(char *bytes);
	concept::packet::UnSubPkt* decodeUnSubPkt(char *bytes);

	concept::packet::PktType decodePktType(char byte);
	std::size_t decodePktLength(char* bytes);

	concept::packet::PktPtr* unmarshalOne(concept::util::CharVector & bytes);
	std::vector<concept::packet::PktPtr> unmarshalAll();
};

} // packet
} // concept

#endif /* BUFFEREDPACKETUNMARSHALLER_H_ */
