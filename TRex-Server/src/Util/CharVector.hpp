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

#ifndef CHARVECTOR_H_
#define CHARVECTOR_H_

#include "../external.hpp"

namespace concept {
namespace util{

typedef boost::shared_ptr<std::vector<char> > CharVectorPtr;
typedef std::vector<char> CharVector;

template <std::size_t N>
class CharArray: public boost::array<char, N> {  };

} // util
} // concept

#endif /* CHARVECTOR_H_ */
