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

#include "TRexServer.h"

using namespace std;

Outbox::Outbox(PubPkt *pkt, set<int> *clients) {
	this->pkt = pkt;
	this->clients = clients;
	pkt->incRefCount();
}

Outbox::~Outbox() {
	if (pkt->decRefCount()) delete pkt;
	delete clients;
}

TRexServer::TRexServer(TRexEngine *parEngine, int parPort, GenericRoutingTable *parRoutingTable, int parQueueSize) {
	engine = parEngine;
	port = parPort;
	routingTable = parRoutingTable;
	queueSize = parQueueSize;
	running = true;
	serverSocket = 0;
	socketsMutex = new pthread_mutex_t;
	pthread_mutex_init(socketsMutex, NULL);
	inputMutex = new pthread_mutex_t;
	pthread_mutex_init(inputMutex, NULL);
	inputCond = new pthread_cond_t;
	pthread_cond_init(inputCond, NULL);
	outputMutex = new pthread_mutex_t;
	pthread_mutex_init(outputMutex, NULL);
	outputCond = new pthread_cond_t;
	pthread_cond_init(outputCond, NULL);
	writingThread = new pthread_t;
	inputReaderThread = new pthread_t;
	pthread_create(writingThread, NULL, trex_server::startSendingThread, (void *) this);
	pthread_create(inputReaderThread, NULL, trex_server::startQueueReader, (void *) this);
	engine->addResultListener(this);
}

TRexServer::~TRexServer() {
	pthread_cond_destroy(inputCond);
	pthread_mutex_destroy(inputMutex);
	pthread_mutex_destroy(socketsMutex);
	delete inputMutex;
	delete inputCond;
	delete socketsMutex;
	while (! outputQueue.empty()) {
		Outbox *outbox = outputQueue.front();
		outputQueue.pop();
		delete outbox;
	}
	while (! inputQueue.empty()) {
		PubPkt *pkt = inputQueue.front();
		inputQueue.pop();
		if (pkt->decRefCount()) delete pkt;
	}
	pthread_mutex_destroy(outputMutex);
	delete outputMutex;
	pthread_cond_destroy(outputCond);
	delete outputCond;
	for (map<int, pthread_t *>::iterator it=readingThreads.begin(); it!=readingThreads.end(); ++it) {
		pthread_t *t = it->second;
		pthread_cancel(*t);
		delete t;
	}
	pthread_cancel(*writingThread);
	delete writingThread;
	pthread_cancel(*inputReaderThread);
	delete inputReaderThread;
}

void TRexServer::startListening() {
	struct sockaddr_in sa;
	memset(&sa, 0, sizeof(struct sockaddr_in));
	sa.sin_family = AF_INET;
	sa.sin_port = htons(port);
	sa.sin_addr.s_addr = htonl(INADDR_ANY);
	// Create the socket and bind it
	serverSocket = socket (AF_INET, SOCK_STREAM, 0);
	if (serverSocket<0) {
		perror("creating the socket");
		exit(-1);
	}
	if (bind(serverSocket, (struct sockaddr *)&sa, sizeof(sa))<0){
		perror("binding");
		exit(-1);
	}
	if (listen(serverSocket, 0)<0) {
		perror("listening");
		exit(-1);
	}
	// Accept incoming requests (main loop)
	while (running) {
		// Accept a new connection
		int clientSocket = accept(serverSocket, NULL, NULL);
		if (clientSocket<0) {
			perror("accepting");
			exit(-1);
		}
		pthread_mutex_lock(socketsMutex);
		int clientId=0;
		while (inputSockets.find(clientId)!=inputSockets.end()) {
			clientId++;
		}
		inputSockets.insert(make_pair(clientId, clientSocket));
		pthread_t *t = new pthread_t;
		readingThreads.insert(make_pair(clientId, t));
		pthread_mutex_unlock(socketsMutex);
		trex_server::Shared *s = (trex_server::Shared *) malloc(sizeof(Shared));
		s->server = this;
		s->clientId = clientId;
		pthread_create(t, NULL, trex_server::startReadingThread, (void *) s);
	}
	close(serverSocket);
}

void TRexServer::stop() {
	running = false;
}

void TRexServer::readFromInputQueue() {
	while (true) {
		pthread_mutex_lock(inputMutex);
		if (inputQueue.empty()) {
			pthread_cond_wait(inputCond, inputMutex);
		}
		if (! running) {
			pthread_mutex_unlock(inputMutex);
			break;
		}
		PubPkt *pkt = inputQueue.front();
		inputQueue.pop();
		pthread_mutex_unlock(inputMutex);
		engine->processPubPkt(pkt);
	}
}

void TRexServer::writeToInputQueue(int clientId) {
	pthread_mutex_lock(socketsMutex);
	int socketId = inputSockets[clientId];
	pthread_mutex_unlock(socketsMutex);
	Unmarshaller unmarshaller;
	while (running) {
		int n;
		// Reads the packet type
		char typeByteArray[1];
		int alreadyRead = 0;
		while (alreadyRead < 1) {
			n = read(socketId, typeByteArray, 1);
			if (n<=0) {
				removeDisconnectedClient(clientId, true);
				break;
			}
			alreadyRead += n;
		}
		PktType pktType = unmarshaller.decodePktType(typeByteArray);
		// Reads the packet length
		char lengthByteArray[4];
		alreadyRead = 0;
		while (alreadyRead < 4) {
			n = read(socketId, lengthByteArray+alreadyRead, 4-alreadyRead);
			if (n<=0) {
				removeDisconnectedClient(clientId, true);
				break;
			}
			alreadyRead += n;
		}
		int length = unmarshaller.decodeInt(lengthByteArray);
		// Reads the packet
		char pktByteArray[length];
		alreadyRead = 0;
		while (alreadyRead < length) {
			n = read(socketId, pktByteArray+alreadyRead, length-alreadyRead);
			if (n<=0) {
				removeDisconnectedClient(clientId, true);
				break;
			}
			alreadyRead += n;
		}
		// Decodes the packet
		if (pktType==PUB_PKT) {
			PubPkt *pkt = unmarshaller.decodePubPkt(pktByteArray);
			pthread_mutex_lock(inputMutex);
			inputQueue.push(pkt);
			pthread_cond_signal(inputCond);
			pthread_mutex_unlock(inputMutex);
		} else if (pktType==SUB_PKT) {
			SubPkt *pkt = unmarshaller.decodeSubPkt(pktByteArray);
			routingTable->installSubscription(clientId, pkt);
		} else if (pktType==JOIN_PKT) {
			JoinPkt *pkt = unmarshaller.decodeJoinPkt(pktByteArray);
			connectToSubscriber(clientId, pkt->getAddress(), pkt->getPort());
			delete pkt;
		} else {
			cout << "Received an unknown packet, probably due to a communication error: disconnecting the client." << endl;
			removeDisconnectedClient(clientId, true);
		}
	}
}

void TRexServer::readFromOutputQueue() {
	Marshaller marshaller;
	while (running) {
		pthread_mutex_lock(outputMutex);
		if (outputQueue.empty()) {
			pthread_cond_wait(outputCond, outputMutex);
		}
		Outbox *out = outputQueue.front();
		outputQueue.pop();
		pthread_mutex_unlock(outputMutex);
		PubPkt *pkt = out->getPkt();
		int size = marshaller.getSize(pkt);
		char * sendArray = marshaller.encode(pkt);
		for (set<int>::iterator it=out->getClients()->begin(); it!=out->getClients()->end(); ++it) {
			int clientId = *it;
			int socketId = -1;
			pthread_mutex_lock(socketsMutex);
			map<int, int>::iterator outputSocketsIt = outputSockets.find(clientId);
			if (outputSocketsIt!=outputSockets.end()) {
				socketId = outputSocketsIt->second;
			}
			pthread_mutex_unlock(socketsMutex);
			if (socketId<0) continue; // No connection to the subscriber
			int n = write(socketId, sendArray, size);
			if (n<0) {
				removeDisconnectedClient(socketId, false);
				continue;
			}
		}
		delete sendArray;
		delete out;
	}
}

void TRexServer::handleResult(set<PubPkt *> &genPkts, double procTime) {
	for (set<PubPkt *>::iterator it=genPkts.begin(); it!=genPkts.end(); ++it) {
		PubPkt *pkt = *it;
		set<int> *subClients = new set<int>;
		routingTable->getMatchingClients(pkt, *subClients);
		if (recursiveTypes.find(pkt->getEventType())!=recursiveTypes.end()) {
			pkt->incRefCount();
			pthread_mutex_lock(inputMutex);
			inputQueue.push(pkt);
			pthread_cond_signal(inputCond);
			pthread_mutex_unlock(inputMutex);
		}
		if (subClients->empty()) {
			delete subClients;
			continue;
		}
		Outbox *out = new Outbox(pkt, subClients);
		pthread_mutex_lock(outputMutex);
		outputQueue.push(out);
		pthread_cond_signal(outputCond);
		pthread_mutex_unlock(outputMutex);
	}
}

void TRexServer::addToRecursiveTypes(int eventId) {
	recursiveTypes.insert(eventId);
}

void TRexServer::connectToSubscriber(int clientId, long address, int port) {
	struct sockaddr_in sa;
	memset(&sa, 0, sizeof(struct sockaddr_in));
	sa.sin_family = AF_INET;
	sa.sin_port = htons(port);
	sa.sin_addr.s_addr = htonl(address);
	int sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock<0) {
		cout << "Error creating the output socket" << endl;
		return;
	}
	if (connect(sock, (struct sockaddr *) &sa, sizeof(sa))<0) {
		cout << "Error while connecting to client" << endl;
		return;
	}
	pthread_mutex_lock(socketsMutex);
	outputSockets.insert(make_pair(clientId, sock));
	pthread_mutex_unlock(socketsMutex);
}

void TRexServer::removeDisconnectedClient(int clientId, bool reading) {
	pthread_mutex_lock(socketsMutex);
	map<int, int>::iterator inputSocketIt=inputSockets.find(clientId);
	if (inputSocketIt!=inputSockets.end()) {
		inputSockets.erase(inputSocketIt);
	}
	if (! reading) {
		map<int, pthread_t *>::iterator readingThreadIt=readingThreads.find(clientId);
		if (readingThreadIt!=readingThreads.end()) {
			pthread_t *t = readingThreadIt->second;
			readingThreads.erase(readingThreadIt);
			pthread_cancel(*t);
			delete t;
		}
	}
	map<int, int>::iterator outputSocketsIt=outputSockets.find(clientId);
	if (outputSocketsIt!=outputSockets.end()) {
		outputSockets.erase(outputSocketsIt);
	}
	pthread_mutex_unlock(socketsMutex);
	if (reading) pthread_exit(NULL);
}

void * trex_server::startQueueReader(void *input) {
	TRexServer *server = (TRexServer *) input;
	server->readFromInputQueue();
	pthread_exit(NULL);
}

void * trex_server::startReadingThread(void *input) {
	Shared *s = (Shared *) input;
	TRexServer *server = s->server;
	int clientId = s->clientId;
	free(s);
	server->writeToInputQueue(clientId);
	pthread_exit(NULL);
}

void * trex_server::startSendingThread(void *input) {
	TRexServer *server = (TRexServer *) input;
	server->readFromOutputQueue();
	pthread_exit(NULL);
}
