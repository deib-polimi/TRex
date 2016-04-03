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

#ifndef ENGINE_H_
#define ENGINE_H_

#include <set>
#include "../Packets/PubPkt.h"
#include "../Packets/RulePkt.h"
#include "ResultListener.h"

class Engine {

public:
  virtual ~Engine() {}

  virtual void processPubPkt(PubPkt* event) = 0;
  virtual void processRulePkt(RulePkt* rule) = 0;
  virtual void addResultListener(ResultListener* resultListener) = 0;
  virtual void removeResultListener(ResultListener* resultListener) = 0;
};

#endif
