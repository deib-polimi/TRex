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

#ifndef CONNECTION_H
#define CONNECTION_H

#include "../external.hpp"
#include "../packet.hpp"
#include "../util.hpp"

#include "ConnectionProxy.hpp"
#include "RequestHandler.hpp"

namespace concept{
namespace connection{

// Represents a single connection with a client.
class Connection: public boost::enable_shared_from_this<Connection> {
public:
#ifdef HAVE_GTREX
  	Connection(boost::asio::io_service& io_service, TRexEngine &tRexEngine, GPUEngine &gtRexEngine, SubscriptionTable &subTable, bool usePing, bool useGPUPar);
#else
	Connection(boost::asio::io_service& io_service, TRexEngine &tRexEngine, SubscriptionTable &subTable, bool usePing, bool useGPUPar);
#endif
	virtual ~Connection(){ }

	// Start the first asynchronous operation for the connection.
	void start();

	// Get the socket associated with the connection.
	boost::asio::ip::tcp::socket& getSocket() {	return socket; }

private:
	// Handle completion of a read operation.
	void handleRead(const boost::system::error_code& error, std::size_t bytes_transferred);

	// Handle completion of a write operation.
	void handleWrite(const boost::system::error_code& error, concept::util::CharVectorPtr);

	void handlePingReceive(const boost::system::error_code& error);
	void handlePingSend(const boost::system::error_code& error);

	void asyncWrite(concept::util::CharVectorPtr bytes);
	void asyncReadSome();

	void startPingReceive();
	void startPingSend();

	void closeSocket();

	std::string printLocalEndpoint();
	std::string printRemoteEndpoint();
	std::string printSocket();

	class ConnectionProxyImpl: public concept::connection::ConnectionProxy{
	private:
		Connection &parent;
	public:
		ConnectionProxyImpl(Connection& parent):parent(parent) { }
		~ConnectionProxyImpl() { }
		void asyncWrite(concept::util::CharVectorPtr bytes) { parent.asyncWrite(bytes); }
		int getLocalPort() { return parent.localPort; }
		int getRemotePort() { return parent.remotePort; }
		std::string getLocalAddress() { return parent.localAddress; }
		std::string getRemoteAddress() { return parent.remoteAddress; }
		std::string localToString() { return parent.printLocalEndpoint(); }
		std::string remoteToString() { return parent.printRemoteEndpoint(); }
		std::string toString() { return parent.printSocket(); }
	};

	// Space reserved for buffer
	static const std::size_t BUFFER_LENGTH= 2048;

	// Heartbeat timeout (in ms). Connection is considered lost when no heartbeat is received
	// by this timeout.
	static const long PING_TIMEOUT= 125000;

	// Heartbeat delay (in ms). A ping is sent to remote endpoint with this
	// periodicity.
	static const long PING_DELAY= 30000;

	// Strand to ensure the connection's handlers are not called concurrently.
	boost::asio::io_service::strand strand;

	// Socket for the connection.
	boost::asio::ip::tcp::socket socket;

	boost::asio::deadline_timer pingReceiveTimer;
	boost::asio::deadline_timer pingSendTimer;
	bool stopTimers;
	bool usePing;

	int localPort;
	int remotePort;
	std::string localAddress;
	std::string remoteAddress;

	// Buffer for incoming data.
	concept::util::CharArray<BUFFER_LENGTH> buffer;
	concept::packet::PacketMarshaller marshaller;

	TRexEngine &tRexEngine;
	
#ifdef HAVE_GTREX
	GPUEngine &gtRexEngine;
#endif

	ConnectionProxyImpl connectionProxy;
	RequestHandler requestHandler;
};

} // namespace connection
} // namespace concept

#endif // CONNECTION_H
