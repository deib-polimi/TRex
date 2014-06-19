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

#ifndef TREXUTILS_H_
#define TREXUTILS_H_

#include "../external.hpp"

namespace concept{
namespace util{

extern bool matches(SubPkt* sub, PubPkt* pub);
extern bool matches(const Constraint& constr, const Attribute& attr);

extern bool equals(SubPkt* pkt1, SubPkt* pkt2);

extern std::string toString(PubPkt* pkt);
extern std::string toString(RulePkt* pkt);
extern std::string toString(SubPkt* pkt);
extern std::string toString(const Attribute& attr);
extern std::string toString(const Constraint& constr);
extern std::string toString(const boost::asio::ip::tcp::socket& socket);
extern std::string toString(const boost::asio::ip::tcp::socket::endpoint_type& endpoint);

} // util
} // concept

#endif /* TREXUTILS_H_ */
