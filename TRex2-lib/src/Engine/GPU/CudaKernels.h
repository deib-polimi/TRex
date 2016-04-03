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

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <set>
#include <map>
#include <iostream>
#include <cuda.h>
#include "GPUProcessorIf.h"
#include "MemoryManager.h"
#include "../../Common/Consts.h"
#include "../../Common/Funs.h"
#include "../CompositeEventGenerator.h"
// #include "../../Common/Definitions.h"

using namespace std;

class CudaKernels {

public:
  /**
   * Get a new intance of this class
   */
  static CudaKernels* getInstance(GPUProcessorIf* processor, MemoryManager* m,
                                  int mmT);

  /**
   * Destructor
   */
  ~CudaKernels();

  /**
   * Sets the processor that is using the kernel
   */
  void setProcessor(GPUProcessorIf* processor);

  /**
   * Decreases the reference count.
   * If the reference count goes to zero, deletes the kernel.
   */
  static void decRefCount();

  /**
   * Sets the parameters
   */
  bool setParameters(map<int, set<GPUParameter*>> parameters);

  /**
   * Sets the aggregates
   */
  void setAggregates(map<int, Aggregate*> aggregates);

  /**
   * Sets the parameters for aggregates
   */
  bool setAggregatesParameters(
      map<int, set<GPUParameter*>> aggregatesParameters);

  /**
   * Sets the parameters for negations
   */
  bool setNegationsParameters(map<int, set<GPUParameter*>> negationsParameters);

  /**
   * Sets the stacks of EventInfo for events and negations
   */
  void setEventsStacks();

  /**
   * Copies the last event to the device memory
   */
  void copyLastEventToDevice(PubPkt* event);

  /**
   * Performs the computation on the device
   */
  void compute();

  /**
   * Retrieves information from the device and fills the set of generated events
   */
  void getGeneratedEvents(set<PubPkt*>& generatedEventsSet);

private:
  /**
   * Sets the stacks of EventInfo for aggregates
   */
  void setAggregateStacks();
  void initAliveChecker();
  uint8_t aliveChecker;
  EventInfoSet* eventsSet;
  Parameter hostParameters[MAX_PARAMETERS_NUM];
  GPUParameter hostComplexParameters[MAX_PARAMETERS_NUM];
  int parametersNum;

  void setAllNegations(int i, int n, negsPasser& negp, int& negSize,
                       int& negMinId);
  void setNegation(int n, passer& negp, parPasser& parp, int& negSize,
                   int& negminId);
  LoopKind prepareForNextLoop(int state, int& size, int& minId, passer& pages,
                              parPasser& parameters,
                              int negs[MAX_NEGATIONS_NUM], int& negsnum,
                              bool& firstWithin);
  float aggResult;
  int mmToken;
  MemoryManager* mm;
  bool checkNegations(EventInfoSet eventsSet);
  void deleteConsumed(EventInfoSet* eventsSet, int size);
  int foots;
  // The instance, used in the case of a singleton implementation
  static CudaKernels* instance;
  // Reference counting variable
  int refCount;
  // Used to add a new event to the matrix of received
  EventInfo* buffer;
  int* resultsSize;
  int* aggResultsSize;
  // Pointer to the result of gpu reduction in host memory
  float* aggRes;
  // Representation of the last event in host memory
  EventInfoSet* lastEvent;
  // Pointers to partial values for aggregates in device memory
  int* aggregatesValPtr;
  // Number of parameters defined for each state
  int parametersSize[MAX_RULE_FIELDS - 1][MAX_PARAMETERS_NUM];
  // Number of parameters defined for each aggregate
  int aggregatesParametersSize[MAX_RULE_FIELDS - 1][MAX_PARAMETERS_NUM];
  int negationsParametersSize[MAX_RULE_FIELDS - 1][MAX_PARAMETERS_NUM];
  // Parameters handler
  GPUProcessorIf* processor;
  // Set of aggregates defined for the rule
  map<int, Aggregate*> aggregates;

  /**
   * Constructor. It is kept private since some implementations require this
   * class to be implemented as a singleton
   */
  CudaKernels(GPUProcessorIf* processor, MemoryManager* m, int mmT);

  /**
   * Compute the value of aggregates.
   * Used indexes contains the events used at each state.
   * Stores the result of each aggregate in the results map.
   */
  float computeAggregates(EventInfoSet& usedEvents,
                          map<int, Aggregate*>& aggregates, int i, OpTree* tree,
                          Attribute* attributes);
  int computeIntValue(EventInfoSet& events, map<int, Aggregate*>& aggregates,
                      OpTree* opTree, Attribute* attributes);
  bool computeBoolValue(EventInfoSet& events, map<int, Aggregate*>& aggregates,
                        OpTree* opTree);
  void computeStringValue(EventInfoSet& events,
                          map<int, Aggregate*>& aggregates, OpTree* opTree,
                          char* result);
  float computeFloatValue(EventInfoSet& events,
                          map<int, Aggregate*>& aggregates, OpTree* opTree,
                          Attribute* attributes);
  bool staticAttributesAdded;
  void addStaticAttributes(CompositeEventTemplate* ceTemplate,
                           Attribute* attributes);
};

#endif
