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

#include "SOEPServer.hpp"

using boost::asio::ip::tcp;
using concept::server::SOEPServer;
using concept::connection::Connection;

#ifdef HAVE_GTREX
SOEPServer::SOEPServer(int port, int thread_pool_size, bool usePingPar, bool useGPUPar) :
			threadNum(thread_pool_size),
			tRexEngine(thread_pool_size),
			gtRexEngine(),
			subTable(),
			acceptor(ioService,tcp::endpoint(tcp::v4(), port)),
			usePing(usePingPar),
			useGPU(useGPUPar),
			newConnection(new Connection(ioService, tRexEngine, gtRexEngine, subTable, usePingPar, useGPUPar))
{
	acceptor.async_accept(newConnection->getSocket(),
			boost::bind(&SOEPServer::handle_accept, this, boost::asio::placeholders::error));			
}
#else
SOEPServer::SOEPServer(int port, int thread_pool_size, bool usePingPar, bool useGPUPar) :
			threadNum(thread_pool_size),
			tRexEngine(thread_pool_size),
			subTable(),
			acceptor(ioService,tcp::endpoint(tcp::v4(), port)),
			usePing(usePingPar),
			useGPU(useGPUPar),
			newConnection(new Connection(ioService, tRexEngine, subTable, usePingPar, useGPUPar))
{
	acceptor.async_accept(newConnection->getSocket(),
			boost::bind(&SOEPServer::handle_accept, this, boost::asio::placeholders::error));
}
#endif

void SOEPServer::run() {
	// Create a pool of threads to run all of the io_services.
	std::vector<boost::shared_ptr<boost::thread> > threads;
	for (std::size_t i = 0; i < threadNum; ++i) {
		boost::shared_ptr<boost::thread> thread(new boost::thread(
				boost::bind(&boost::asio::io_service::run, &ioService)));
		threads.push_back(thread);
	}

	// Wait for all threads in the pool to exit.
	for (std::size_t i = 0; i < threads.size(); ++i){
		threads[i]->join();
	}
}

void SOEPServer::stop() {
	ioService.stop();
}

void SOEPServer::handle_accept(const boost::system::error_code& error) {

	if (!error) newConnection->start();
#ifdef HAVE_GTREX
	newConnection.reset(new Connection(ioService, tRexEngine, gtRexEngine, subTable, usePing, useGPU));
#else
	newConnection.reset(new Connection(ioService, tRexEngine, subTable, usePing, useGPU));
#endif
	acceptor.async_accept(newConnection->getSocket(),
			boost::bind(&SOEPServer::handle_accept, this, boost::asio::placeholders::error));
}
