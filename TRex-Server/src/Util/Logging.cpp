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

#include "Logging.hpp"

using namespace concept::util;

namespace logging = boost::log;
namespace sinks = boost::log::sinks;
namespace src = boost::log::sources;
namespace attrs = boost::log::attributes;
namespace keywords = boost::log::keywords;
namespace expr = boost::log::expressions;

#	if BOOST_VERSION < 105500
namespace boost {
	using empty_deleter = log::empty_deleter;
}
#	endif

// init static logger
src::severity_logger< severity_level > Logging::logger;

void Logging::init(){
	// Construct the sink (frontend)
	typedef sinks::synchronous_sink< sinks::text_ostream_backend > text_sink;
	boost::shared_ptr< text_sink > pSink = boost::make_shared< text_sink >();

	// Get a thread-safe pointer to sink backend
	text_sink::locked_backend_ptr pBackend = pSink->locked_backend();

	// Automatically flush streams after each record
	pBackend->auto_flush(true);

	// Add a stream to log to console
	boost::shared_ptr< std::ostream > pConsoleStream(&std::cout, boost::empty_deleter());
	pBackend->add_stream(pConsoleStream);

	// Add a stream to log to file
	boost::shared_ptr< std::ofstream > pFileStream(new std::ofstream("session.log"));
	pBackend->add_stream(pFileStream);

	// Register the sink in the logging core
	logging::core::get()->add_sink(pSink);

	// And a global timestamp attribute
// 	boost::shared_ptr< logging::attribute > pTimeStamp(new attrs::local_clock());
//  	logging::core::get()->add_global_attribute("TimeStamp", pTimeStamp);
	logging::core::get()->add_global_attribute("TimeStamp", attrs::local_clock());
	// Describe how to format logs
	pSink->set_formatter(
	    expr::format("%1% %2%")
		% expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%d.%m.%Y %H:%M:%S")
		% expr::smessage
	);


	// Add a filter on severity level:
	// Write all records with "info" severity or higher
//  	pSink->set_filter(
//  		logging::trivial::severity >= logging::trivial::info
//  	);
}
