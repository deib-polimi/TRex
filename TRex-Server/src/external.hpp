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

#ifndef EXTERNAL_H_
#define EXTERNAL_H_

// Config.h
#include "../config.h"

// Boost
#include <boost/array.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/variant.hpp>
#include <boost/phoenix/bind.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared_object.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/support/date_time.hpp>
#if BOOST_VERSION >= 105500
#include <boost/utility/empty_deleter.hpp>
#else
#include <boost/log/utility/empty_deleter.hpp>
#endif


// C++ Standard Library
#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

// C Standard Library
#include <unistd.h>

// TRex
#include <Common/Consts.h>
#include <Engine/TRexEngine.h>
#ifdef HAVE_GTREX
#include <Engine/GPU/GPUEngine.h>
#endif
#include <Marshalling/Marshaller.h>
#include <Marshalling/Unmarshaller.h>
#include <Packets/PubPkt.h>
#include <Packets/RulePkt.h>
#include <Packets/RulePktValueReference.h>

#endif /* EXTERNAL_H_ */
