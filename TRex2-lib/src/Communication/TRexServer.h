//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//

#ifndef TREXSERVER_H_
#define TREXSERVER_H_

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <queue>
#include <map>
#include "../Packets/PubPkt.h"
#include "../Packets/SubPkt.h"
#include "GenericRoutingTable.h"
#include "../Engine/TRexEngine.h"
#include "../Engine/ResultListener.h"
#include "../Marshalling/Marshaller.h"
#include "../Marshalling/Unmarshaller.h"
#include "../Common/Consts.h"

/**
 * Outbox: contains a message and the set of clients it has to be delivered to.
 */
class Outbox {
public:
	/**
	 * Constructor
	 */
	Outbox(PubPkt *pkt, std::set<int> *clients);

	/**
	 * Destructor
	 */
	virtual ~Outbox();

	/**
	 * Returns the stored packet
	 */
	PubPkt * getPkt() {
		return pkt;
	}

	/**
	 * Returns the set of clients the packet has to be sent to
	 */
	std::set<int> * getClients() {
		return clients;
	}

private:
	PubPkt *pkt;
	std::set<int> *clients;
};

/**
 * This is the T-Rex server.
 * It handles the communication with connected clients by:
 * 1) Managing their subscriptions
 * 2) Sending received events to the T-Rex engine
 * 3) Sending generated events to subscribed clients
 *
 * NB In the current implementation it does not allow users to dynamically
 * install/remove rules. Rules must be already installed in the TRexEngine
 * passed to the constructor, and the finalize() must be called on it to
 * start event generation.
 */
class TRexServer : public ResultListener {
public:

	/**
	 * Creates a new server, listening on the given port.
	 * engine is used to perform processing.
	 * routingTable is used to store subscriptions.
	 * queueSize represents the maximum size of the queue for input events.
	 */
	TRexServer(TRexEngine *engine, int port, GenericRoutingTable *routingTable, int queueSize);

	/**
	 * Destructor.
	 */
	virtual ~TRexServer();

	/**
	 * Start listening for client connections.
	 */
	void startListening();

	/**
	 * Stops the server.
	 */
	void stop();

	/**
	 * Reads from the input queue and sends each received message to the T-Rex engine.
	 */
	void readFromInputQueue();

	/**
	 * Reads from the socket of the given client and writes to the input queue.
	 */
	void writeToInputQueue(int clientId);

	/**
	 * Reads from the output queue, serializes and sends messages.
	 */
	void readFromOutputQueue();

	/**
	 * Handle results coming from the T-Rex engine.
	 * Inherited from ResultListener.
	 */
	void handleResult(std::set<PubPkt *> &genPkts, double procTime);

	/**
	 * Adds the eventId to the set of event type that need to be re-submitted
	 * to the engine for further processing once they have been detected.
	 */
	void addToRecursiveTypes(int eventId);

private:
	TRexEngine *engine;													// The processing engine
	GenericRoutingTable *routingTable;					// The routing table used to store subscriptions
	int queueSize;															// The maximum size of the queue for input events
	std::queue<PubPkt *> inputQueue;						// Input queue for received events
	pthread_mutex_t *socketsMutex;							// Mutex to access the input and output socket maps
	std::map<int, int> inputSockets;						// Client id -> input socket for the client
	std::map<int, int> outputSockets;						// Client id -> output socket for the client
	std::queue<Outbox *> outputQueue;						// Output queue with the messages for clients
	int serverSocket;														// Socket used to listen for new connections
	bool running;																// Boolean variable used to stop the server (through the stop() method)
	int port;																		// The port the server listen to
	std::set<int> recursiveTypes;								// Types of events that need to be re-submitted to the engine for further processing

	pthread_cond_t *inputCond;									// Condition variable for the input queue
	pthread_mutex_t *inputMutex;								// Mutex for the input queue
	pthread_mutex_t *outputMutex;								// Mutex for the output queues
	pthread_cond_t *outputCond;									// Condition variable for the output queues
	std::map<int, pthread_t *> readingThreads;	// Client id -> thread reading information coming from the client
	pthread_t *writingThread;										// Thread writing information to the client
	pthread_t *inputReaderThread;								// Thread reading information from the input queue and passing it to the T-Rex engine

	/**
	 * Connects to the remote subscriber having the given clientId.
	 * The subscriber is supposed to be listening on the given address and port.
	 */
	void connectToSubscriber(int clientId, long address, int port);

	/**
	 * Removes the client and all associated data (i.e. mutexes, condition variables, and sockets)
	 * reading==true if the function is called within the readingThread, reading==false otherwise
	 */
	void removeDisconnectedClient(int clientId, bool reading);

};

/**
 * Data structures and functions used by threads
 */
namespace trex_server {

/**
 * Shared memory used to pass parameters to threads.
 */
typedef struct SharedStruct {
	TRexServer *server;
	int clientId;
} Shared;

/**
 * Start the thread to read from the queue of input events and sends them to the T-Rex engine.
 */
void * startQueueReader(void *input);

/**
 * Start a new thread to handle messages coming from a connected client.
 * Each message is appended to the input queue, or dropped if the queue is full.
 */
void * startReadingThread(void *input);

/**
 * Start a new thread to send messages to connected clients.
 */
void * startSendingThread(void *input);
}

#endif /* TREXSERVER_H_ */
