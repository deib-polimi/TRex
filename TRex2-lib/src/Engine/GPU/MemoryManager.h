//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Daniele Rogora
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

#ifndef MEMORYMANAGER_H
#define MEMORYMANAGER_H

#include <cuda.h>
#include "../../Common/Consts.h"
#include "../../Common/Structures.h"
#include <iostream>
#include <map>

using namespace std;

/***
 * This is the memory manager for the CUDA engine. Only one instance is created
 * that is shared among all the GPUProcessors created by the GPUEngine.
 */
class MemoryManager {

public:
  /**
   * Constructor, allocates the memory spaces on the GPU and on the HOST
   */
  MemoryManager();

  /**
   * Destructor
   */
  virtual ~MemoryManager();

  /**
   * Must be called by a CudaKernel in order to start using
   * the GPU memory and the memory manager services;
   * returns a unique int representing the subscriber
   */
  int subscribe(int seqLen, int aggrNum, int negNum, int paramNum);

  /**
   * Must be called by a CudaKernel when it stops using the GPU;
   * releases the resources allocated
   */
  void unsubscribe(int id, int seqLen, int negsNum, int aggrNum);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the result int on the device
   */
  int* getResPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the alive uint8_t array on the device
   */
  uint8_t* getAlivePtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the maxTS int array on the device
   */
  int* getSeqMaxPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the currentIndex int on the device
   */
  int* getResultsSizePtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the states Parameter array on the device
   */
  GPUParameter* getParameterPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the aggregates Parameter array on the device
   */
  GPUParameter* getAggregateParameterPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the negations Parameter array on the device
   */
  GPUParameter* getNegationsParameterPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel,
   * the state (stack number)
   * Returns the indexes of the pages representing
   * the specified state of the specified CudaKernel
   */
  void getPageIndexes(int id, int state, passer& res);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the EventInfoSet representing
   * the previousResults array on the device
   */
  EventInfoSet* getPrevResultsPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Returns the pointer to the EventInfoSet representing
   * the currentResults array on the device
   */
  EventInfoSet* getCurrentResultsPtr(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Swaps the previousResults and currentResults pointers
   */
  void swapPointers(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Restores the previousResults and currentResults
   * pointers to the original state
   */
  void restorePointers(int id);

  /**
   * Requires the identification token of the calling CudaKernel
   * Swaps the previousResults and currentResults pointers
   */
  void setEventsStacks(int id, int s, int minId, int maxId, StateType stype);

  /**
   * Requires the identification token of the calling CudaKernel,
   * the state (stack number)
   * Returns the indexes of the pages representing
   * the specified negation of the specified CudaKernel
   */
  void getNegationsIndexes(int id, int state, passer& res);

  void getAllNegationsIndexes(int id, int index, int nNum, negsPasser& res);

  /**
   * Requires the identification token of the calling CudaKernel,
   * the state (stack number)
   * Returns the indexes of the pages representing the specified
   * aggregate of the specified CudaKernel
   */
  void getAggregatesIndexes(int id, int state, passer& res);

  void getHostAggregatesIndexes(int id, int state, passer& res);

  /**
   * Adds ev to the stack stack of the type kind,
   * in position index, for the CudaKernel id
   * Copies asyncronously on the fly if the corresponding page on the GPU is
   * already used by id, else ev is only added in the HOST memory,
   * waiting for the terminator
   */
  void addEventToStack(int id, EventInfo ev, int stack, int index,
                       StateType type);

  /**
   * Returns the event from the rule id, stack stack and index index;
   * this is only for STATE events, so no AGG nor NEG.
   */
  EventInfo getEvent(int id, int stack, int index);

  /**
   * Removes the STATE event from the rule id, stack stack and index index
   * It operates on the HOST memory only, but it invalidates
   * the corresponding page on the GPU
   */
  void cleanEventStack(int id, int stack, int idx, int& highIndex);

  /**
   * Returns a pointer to the negations in the device memory for CudaKernel id
   */
  GPUNegation* getNegationsPtr(int id);

  /*
   * Returns a pointer to the float array used to compute
   * aggregates used by rule id
   */
  float* getAggregatesValPtr(int id);

  /*
   * Returns the event in position index from stack stack for the given type
   */
  EventInfo getEventFromStack(int id, int stack, int index, StateType type);

private:
  /*
   * Returns true if page page is owned by rule id
   */
  bool ruleOwns(int id, int page);

  /*
   * Returns the index of the first chunk of memory available
   * for a new assignment
   */
  int getFreeChunk(int id);

  /**
   * Sets page page on the GPU as outdated, not valid
   */
  void invalidatePage(int id, int page);

  /**
   * Copies page page to the device memory and updates informations
   * about memory pages on GPU
   */
  void copyPageToDevice(int id, int page);
  void setEvent(int id, int stack, int index, EventInfo ev);
  bool isOnGpu(int id, int page);
  uint32_t getPage(int id, int s, int offset, int minId, StateType stype);
  workspace workspaces[MAX_CONCURRENT_RULES];
  EventInfo* h_base_address;
  EventInfo* base_address;
  int d_chunksnum;
  uint64_t chunksnum;
  bool swapped[MAX_RULE_NUM];
  bool inUse[MAX_RULE_NUM];
  chunk* chunks;
  uint64_t totalMem;
  uint64_t ruleFixedSize;
  uint64_t workspaceFixedSize;
  uint64_t usedMem;
  ruleMemoryStruct rules[MAX_RULE_NUM];
  bool notified;
};

#endif // MEMORYMANAGER_H
