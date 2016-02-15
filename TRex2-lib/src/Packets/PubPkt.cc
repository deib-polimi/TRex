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

#include "PubPkt.h"

using namespace std;

#if LOG == 1
int PubPkt::count = 0;
#endif

PubPkt::PubPkt(int parEventType, Attribute* parAttributes,
               int parAttributesNum) {
  eventType = parEventType;
  attributesNum = parAttributesNum;
  attributes = new Attribute[attributesNum];
  for (int i = 0; i < attributesNum; i++) {
    attributes[i] = parAttributes[i];
    string name = attributes[i].name;
    contentMap.insert(make_pair(name, i));
  }
  referenceCount = 1;
#if MP_MODE == MP_LOCK
  mutex = new pthread_mutex_t;
  pthread_mutex_init(mutex, NULL);
#endif
#if LOG == 1
  count++;
#endif
}

PubPkt::PubPkt(const PubPkt& pkt) {
  eventType = pkt.eventType;
  attributesNum = pkt.attributesNum;
  attributes = new Attribute[attributesNum];
  for (int i = 0; i < attributesNum; i++)
    attributes[i] = pkt.attributes[i];
  timeStamp = pkt.timeStamp;
  referenceCount = 1;
}

PubPkt::~PubPkt() {
  delete[] attributes;
#if MP_MODE == MP_LOCK
  pthread_mutex_destroy(mutex);
  delete mutex;
#endif
#if LOG == 1
  count--;
  if (count == 0)
    cout << "* All messages have been successfully deleted *" << endl;
#endif
}

//#if MP_MODE == MP_COPY
PubPkt* PubPkt::copy() {
  PubPkt* copy = new PubPkt(eventType, attributes, attributesNum);
  copy->timeStamp = timeStamp;
  return copy;
}
//#endif

void PubPkt::setCurrentTime() {
  TimeMs currentTime;
  timeStamp = currentTime;
}

void PubPkt::incRefCount() {
#if MP_MODE == MP_LOCK
  pthread_mutex_lock(mutex);
#endif
  referenceCount++;
#if MP_MODE == MP_LOCK
  pthread_mutex_unlock(mutex);
#endif
}

bool PubPkt::decRefCount() {
  bool returnValue;
#if MP_MODE == MP_LOCK
  pthread_mutex_lock(mutex);
#endif
  returnValue = (--referenceCount == 0);
#if MP_MODE == MP_LOCK
  pthread_mutex_unlock(mutex);
#endif
  return returnValue;
}

bool PubPkt::getAttributeIndexAndType(char* name, int& index, ValType& type) {
  map<string, int>::iterator it = contentMap.find(name);
  if (it == contentMap.end()) {
    return false;
  }
  index = it->second;
  type = attributes[index].type;
  return true;
}

int PubPkt::getIntAttributeVal(int index) { return attributes[index].intVal; }

float PubPkt::getFloatAttributeVal(int index) {
  return attributes[index].floatVal;
}

bool PubPkt::getBoolAttributeVal(int index) {
  return attributes[index].boolVal;
}

void PubPkt::getStringAttributeVal(int index, char* result) {
  strcpy(result, attributes[index].stringVal);
}
