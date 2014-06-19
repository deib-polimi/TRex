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

#include "Connection.hpp"

using concept::connection::Connection;
using namespace concept::packet;
using namespace concept::util;
using namespace std;

#ifdef HAVE_GTREX
Connection::Connection(boost::asio::io_service& io_service, TRexEngine &tRexEngine, GPUEngine &gtRexEngine, SubscriptionTable &subTable, bool usePingPar, bool useGPUPar)
: strand(io_service),
  socket(io_service),
  pingReceiveTimer(io_service),
  pingSendTimer(io_service),
  usePing(usePingPar),
  stopTimers(false),
  tRexEngine(tRexEngine),
  gtRexEngine(gtRexEngine),
  connectionProxy(*this),
  requestHandler(tRexEngine, gtRexEngine, connectionProxy, subTable, useGPUPar)
{ }
#else
Connection::Connection(boost::asio::io_service& io_service, TRexEngine &tRexEngine, SubscriptionTable &subTable, bool usePingPar, bool useGPUPar)
: strand(io_service),
  socket(io_service),
  pingReceiveTimer(io_service),
  pingSendTimer(io_service),
  usePing(usePingPar),
  stopTimers(false),
  tRexEngine(tRexEngine),
  connectionProxy(*this),
  requestHandler(tRexEngine, connectionProxy, subTable, useGPUPar)
{ }
#endif
void Connection::start()
{
	localPort= socket.local_endpoint().port();
	remotePort= socket.remote_endpoint().port();
	localAddress= socket.local_endpoint().address().to_string();
	remoteAddress= socket.remote_endpoint().address().to_string();

	LOG(info) << "Connection accepted from " << printRemoteEndpoint();

	asyncReadSome();
	if (usePing) {
		startPingReceive();
		startPingSend();
	}
}

void Connection::handleRead(const boost::system::error_code& error, std::size_t bytes_transferred) {
	if (!error) {
		if (usePing) {
			pingReceiveTimer.cancel();
		}
		requestHandler.handleRequest<BUFFER_LENGTH>(buffer, bytes_transferred);
		asyncReadSome();
	} else {
		LOG(warning) << "Connection error while receiving from " << printRemoteEndpoint();
		closeSocket();
		stopTimers= true;
		if (usePing) {
			pingReceiveTimer.cancel();
			pingSendTimer.cancel();
		}
	}
}

void Connection::handleWrite(const boost::system::error_code& error, CharVectorPtr buffer){
	if (!error){
		if (usePing) {
			pingSendTimer.cancel();
		}
	}
	else {
		LOG(warning) << "Connection error while sending to " << printRemoteEndpoint();
		closeSocket();
		stopTimers= true;
		if (usePing) {
			pingReceiveTimer.cancel();
			pingSendTimer.cancel();
		}
	}
}

void Connection::handlePingReceive(const boost::system::error_code& error){
	if (error == boost::asio::error::operation_aborted){
		// something has been received from endpoint before timeout expiration:
		// this means that remote endpoint is still alive and we can restart the timer
		if (usePing) {
			startPingReceive();
		}
	}
	else if (stopTimers){
		// one of the handlers associated with the socket detected a connection
		// problem and asked me to exit
	}
	else {
		// nothing was received until timeout expiration:
		// remote endpoint is probably dead, close connection
		LOG(warning) << "Ping timeout expired for " << printRemoteEndpoint();
		closeSocket();
	}
}

void Connection::handlePingSend(const boost::system::error_code& error){
	if (error == boost::asio::error::operation_aborted){
		// something has already been sent to endpoint before timeout expiration:
		// we can restart the timer without sending a ping
		if (usePing) {
			startPingSend();
		}
	}
	else if (stopTimers){
		// one of the handlers associated with the socket detected a connection
		// problem and asked me to exit
	}
	else {
		if (usePing) {
			// it's time to send a ping
			PingPkt* ping= new PingPkt();
			CharVectorPtr bytes= marshaller.marshal(PktPtr(ping));
			delete ping;
			asyncWrite(bytes);

			// restart the timer
			startPingSend();
		}
	}
}

void Connection::startPingReceive(){
	pingReceiveTimer.expires_from_now(boost::posix_time::millisec(PING_TIMEOUT));
	pingReceiveTimer.async_wait(boost::bind(
			&Connection::handlePingReceive,
			shared_from_this(),
			boost::asio::placeholders::error));
}

void Connection::startPingSend(){
	pingSendTimer.expires_from_now(boost::posix_time::millisec(PING_DELAY));
	pingSendTimer.async_wait(boost::bind(
			&Connection::handlePingSend,
			shared_from_this(),
			boost::asio::placeholders::error));
}

void Connection::asyncWrite(CharVectorPtr bytes){
	if (socket.is_open()) {
		boost::asio::async_write(socket, boost::asio::buffer(*bytes),
				strand.wrap(boost::bind(
						&Connection::handleWrite,
						shared_from_this(),
						boost::asio::placeholders::error,
						// the buffer is also passed so that the shared_ptr's life is
						// extended until the handler will be called
						bytes)));
	}
}
void Connection::asyncReadSome(){
	if (socket.is_open()) {
		socket.async_read_some(boost::asio::buffer(buffer),
				strand.wrap(
						boost::bind(&Connection::handleRead, shared_from_this(),
								boost::asio::placeholders::error,
								boost::asio::placeholders::bytes_transferred)));
	}
}

void Connection::closeSocket(){
	if (socket.is_open()) {
		try {
			socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
			socket.close();
			LOG(info) << "Closed socket " << printSocket();
		} catch (boost::system::system_error e) {
			LOG(warning) << "Error in closing socket " << printSocket();
		}
	}
}

std::string Connection::printSocket(){
	stringstream ss;
	ss << "{local=" << printLocalEndpoint() << ", remote=" << printRemoteEndpoint() << "}";
	return ss.str();
}

std::string Connection::printLocalEndpoint(){
	stringstream ss;
	ss << localAddress << "/" << localPort;
	return ss.str();
}

std::string Connection::printRemoteEndpoint(){
	stringstream ss;
	ss << remoteAddress << "/" << remotePort;
	return ss.str();
}
