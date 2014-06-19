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

#ifndef LOGGING_H_
#define LOGGING_H_

#include "../external.hpp"

namespace concept{
namespace util{

enum severity_level
{
	debug,
	info,
	warning,
	error,
	fatal
};

// The formatting logic for the severity level
template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
    std::basic_ostream< CharT, TraitsT >& strm, severity_level lvl)
{
    static const char* const str[] =
    {
        "debug",
        "info",
        "warning",
        "error",
        "fatal"
    };
    if (static_cast< std::size_t >(lvl) < (sizeof(str) / sizeof(*str)))
        strm << str[lvl];
    else
        strm << static_cast< int >(lvl);
    return strm;
}

#undef LOG // Undefine T-Rex LOG macro
#define LOG(lvl) BOOST_LOG_SEV(::concept::util::Logging::logger, lvl)

class Logging {
public:

	static boost::log::sources::severity_logger< severity_level > logger;
	static void init();
};

} // util
} // concept

#endif /* LOGGING_H_ */
