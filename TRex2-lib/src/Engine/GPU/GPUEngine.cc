//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Daniele Rogora
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

#include "GPUEngine.h"
#include <sys/time.h>

void GPUEngine::processRulePkt(RulePkt* rule) {
  setRecursionNeeded(rule);
  if (processors->size() + 1 > MAX_RULE_NUM) {
    cout << "Max number of rules for GPU engine reached; exiting" << endl;
    exit(-1);
  }
  GPUProcessor* processor = new GPUProcessor(rule, mm);
  processor->init();
  processors->push_back(processor);
}

GPUEngine::GPUEngine() {
  generatedEvents.clear();
  // cout << "GPU engine init..." << endl;
  processors = new vector<GPUProcessor*>();
  mm = new MemoryManager();
  recursionNeeded = false;
  // cout << "... done" << endl;
}

GPUEngine::~GPUEngine() {
  for (vector<GPUProcessor*>::iterator it = processors->begin();
       it != processors->end(); ++it) {
    GPUProcessor* processor = *it;
    delete processor;
  }
  delete processors;
  delete mm;
}

void GPUEngine::setRecursionNeeded(RulePkt* pkt) {
  if (recursionNeeded == true)
    return;

  for (int i = 0; i < pkt->getPredicatesNum(); i++)
    inputEvents.insert(pkt->getPredicate(i).eventType);
  outputEvents.insert(pkt->getCompositeEventTemplate()->getEventType());

  for (std::set<int>::iterator it = inputEvents.begin();
       it != inputEvents.end(); ++it) {
    if (outputEvents.find(*it) != outputEvents.end()) {
      recursionNeeded = true;
      return;
    }
  }
}

void GPUEngine::processPubPkt(PubPkt* event, bool recursion) {
  if (recursion == false)
    recursionDepth = 0;
  else
    recursionDepth++;

  set<PubPkt*> result;
  timeval tValStart, tValEnd;
  gettimeofday(&tValStart, NULL);

  for (vector<GPUProcessor*>::iterator it = processors->begin();
       it != processors->end(); ++it) {
    GPUProcessor* processor = *it;
#if MP_MODE == MP_COPY
    processor->processEvent(event->copy(), result);
#elif MP_MODE == MP_LOCK
    processor->processEvent(event, result);
#endif
  }
  gettimeofday(&tValEnd, NULL);
  double duration = (tValEnd.tv_sec - tValStart.tv_sec) * 1000000 +
                    tValEnd.tv_usec - tValStart.tv_usec;
  // Notifies results to listeners
  for (set<ResultListener*>::iterator it = resultListeners.begin();
       it != resultListeners.end(); ++it) {
    ResultListener* listener = *it;
    listener->handleResult(result, duration);
  }

  for (set<PubPkt*>::iterator it = result.begin(); it != result.end(); ++it) {
    PubPkt* pkt = *it;
    if (recursionNeeded && recursionDepth < MAX_RECURSION_DEPTH)
      processPubPkt(pkt->copy(), true);
    if (pkt->decRefCount()) {
      delete pkt;
    }
  }
  if (event->decRefCount())
    delete event;
}

void GPUEngine::processPubPkt(PubPkt* event) {
  return processPubPkt(event, false);
}
