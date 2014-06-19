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

#ifndef SOEPSERVER_H
#define SOEPSERVER_H

#include "../external.hpp"
#include "../connection.hpp"
#include "../util.hpp"

namespace concept{
namespace server{

using concept::connection::SubscriptionTable;

class SOEPServer
{
public:

  SOEPServer(int port, int thread_pool_size, bool usePing, bool useGPUPar);

  // Run the server's io_service loop.
  void run();

  // Stop the server.
  void stop();

  TRexEngine& getEngine(){ return tRexEngine; }
  
#ifdef HAVE_GTREX
  GPUEngine& getgEngine(){ return gtRexEngine; }
#endif

  const static int DEFAULT_PORT= 50254;

private:
  // Handle completion of an asynchronous accept operation.
  void handle_accept(const boost::system::error_code& e);

  // The number of threads that will call io_service::run().
  std::size_t threadNum;

  TRexEngine tRexEngine;
  
#ifdef HAVE_GTREX
  GPUEngine gtRexEngine;
#endif
  
  SubscriptionTable subTable;

  bool usePing;
  
  bool useGPU;

  // The io_service used to perform asynchronous operations.
  boost::asio::io_service ioService;

  // Acceptor used to listen for incoming connections.
  boost::asio::ip::tcp::acceptor acceptor;

  // The next connection to be accepted.
  boost::shared_ptr<concept::connection::Connection> newConnection;
};

} // server
} // concept

#endif //SOEPSERVER_H
