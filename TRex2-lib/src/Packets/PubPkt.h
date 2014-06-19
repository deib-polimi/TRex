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

#ifndef PUBPKT_H_
#define PUBPKT_H_

#include "../Common/Consts.h"
#include "../Common/Structures.h"
#include "../Common/TimeMs.h"
#include <map>
#include <string.h>
#include <pthread.h>

/**
 * This class represents a publication packet, which is composed by a type,
 * a set of attributes (name-value couples) and a time stamp (its publication time).
 */
class PubPkt {
public:

	/**
	 * Constructor with parameters
	 */
	PubPkt(int parEventType, Attribute *parAttributes, int parAttributesNum);

	/**
	 * Copy constructor
	 */
	PubPkt(const PubPkt &pkt);

	/**
	 * Destructor
	 */
	virtual ~PubPkt();

//#if MP_MODE == MP_COPY
	/**
	 * Creates an exact copy of the packet
	 */
	PubPkt * copy();
//#endif

	/**
	 * Set time (useful for test evaluation)
	 */
	void setCurrentTime();

	void setTime(TimeMs parTimeStamp) { timeStamp = parTimeStamp; }

	/**
	 * Getter methods
	 */
	int getEventType() { return eventType; }

	TimeMs getTimeStamp() { return timeStamp; }

	int getAttributesNum() { return attributesNum; }

	Attribute getAttribute(int attNum) { return attributes[attNum]; }

	/**
	 * Fills index and type with the index and type of the attribute having the given name, if any.
	 * Otherwise returns false.
	 */
	bool getAttributeIndexAndType(char *name, int &index, ValType &type);

	/**
	 * Returns the value of the attribute at the given index.
	 * Requires index to be a valid attribute index.
	 * Requires the type of the attribute to be INT.
	 */
	int getIntAttributeVal(int index);

	/**
	 * Returns the value of the attribute at the given index.
	 * Requires index to be a valid attribute index.
	 * Requires the type of the attribute to be FLOAT.
	 */
	float getFloatAttributeVal(int index);

	/**
	 * Returns the value of the attribute at the given index.
	 * Requires index to be a valid attribute index.
	 * Requires the type of the attribute to be BOOL.
	 */
	bool getBoolAttributeVal(int index);

	/**
	 * Returns the value of the attribute at the given index.
	 * The result is stored in the memory pointed by the result parameter.
	 * Requires index to be a valid attribute index.
	 * Requires the type of the attribute to be STRING.
	 */
	void getStringAttributeVal(int index, char *result);

	/**
	 * Increases reference count
	 * - Synchronized method -
	 */
	void incRefCount();

	/**
	 * Decreases reference count and return true if it becomes 0 (synchronized method)
	 * - Synchronized method -
	 */
	bool decRefCount();

private:

#if LOG == 1
	static int count;
#endif

#if MP_MODE == MP_LOCK
	pthread_mutex_t *mutex;	// Mutex for synchronized methods
#endif

	uint16_t eventType;													// Type of the event
	TimeMs timeStamp;												// Time stamp of the packet
	Attribute *attributes;									// Set of attributes (name-value couples)
	uint16_t attributesNum;											// Number of attributes
	uint16_t referenceCount;											// Number of references to the packet
	std::map<std::string, int> contentMap;	// Name -> index of the attribute holding that name
};

#endif
