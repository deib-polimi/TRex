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

#ifndef GPUPROCESSORIF_H_
#define GPUPROCESSORIF_H_

#include "../../Common/Consts.h"
#include "../../Packets/PubPkt.h"
#include "../../Packets/RulePkt.h"

/**
 * This represents the part of a GPUProcessor that is exposed to the Cuda
 * Kernels.
 */
class GPUProcessorIf {
public:
  /**
   * Returns the minimum element in the given stack having
   * a time stamp greater than minTimestamp.
   * If agg==true, stack represents an aggregate stack.
   */
  virtual int getMinValidElement(int stack, uint64_t minTimestamp,
                                 StateType t) = 0;

  /**
   * Returns the maximum element in the given stack having
   * a time stamp greater than maxTimestamp.
   * If agg==true, stack represents an aggregate stack.
   */
  virtual int getMaxValidElement(int stack, uint64_t maxTimestamp,
                                 StateType t) = 0;

  /**
   * Returns the low index for the given stack.
   * If agg==true, stack represents an aggregate stack.
   */
  virtual int getLowIndex(int stack, StateType t) = 0;

  /**
   * Returns the high index for the given stack.
   * If agg==true, stack represents an aggregate stack.
   */
  virtual int getHighIndex(int stack, StateType t) = 0;

  /**
   * Returns the time stamp of the event stored in the given stack,
   * at the given index.
   * If agg==true, stack represents an aggregate stack.
   */
  virtual uint64_t getTimestamp(int stack, int index, StateType t) = 0;

  virtual CompKind getCompKind(int state) = 0;
  virtual int getCompositeEventId() = 0;
  virtual int getAttributesNum() = 0;
  virtual int getStaticAttributesNum() = 0;
  virtual uint64_t getWin(int state) = 0;
  virtual int getSequenceLen() = 0;
  virtual void getConstraint(int state, int num, int& idx, Op& o, int& val) = 0;
  virtual int getAggregatesNum() = 0;
  virtual uint64_t getMaxTimestamp(int stack, StateType t) = 0;
  virtual bool getAttributeIndexAndType(int state, char* name, int& index,
                                        ValType& type, StateType stype) = 0;
  virtual int getNumParam() = 0;
  virtual bool isConsuming(int state) = 0;
  virtual void remove(EventInfo ev, int stack) = 0;
  virtual int getNegationsNum() = 0;
  virtual Negation* getNegation(int num) = 0;
  virtual int getNegsSize(int negIndex) = 0;
  virtual CompositeEventTemplate* getCompositeEvent() = 0;
  virtual int getNumParamAggregates() = 0;
  virtual int getNumParamNegations() = 0;
  virtual int getRefersTo(int state) = 0;
};

#endif
