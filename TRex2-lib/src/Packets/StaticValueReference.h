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

#ifndef STATICVALUEREFERENCE_H_
#define STATICVALUEREFERENCE_H_

#include <stdlib.h>
#include <string.h>
#include "../Common/OpValueReference.h"
#include "../Common/Consts.h"

/**
 * A StaticValueReference extends the OpValueReference class and defines
 * each reference to value by pointing to an attribute or aggregate in the
 * RulePkt.
 */
class StaticValueReference : public OpValueReference {

public:
  /**
   * Constructor for a reference to a normal state attribute.
   * Parameters: index in the packet, and name of the attribute.
   */
  StaticValueReference(float pValue);

  /**
   * Constructor for a reference to an aggregate state.
   */
  StaticValueReference(int pValue);

  StaticValueReference(bool pValue);

  StaticValueReference(char* pValue);

  /**
   * Destructor
   */
  virtual ~StaticValueReference();

  /**
   * Creates an exact copy of the data structure
   */
  OpValueReference* dup();

  ValType getType();

  int getIntValue();

  float getFloatValue();

  bool getBoolValue();

  void getStringValue(char* pValue);

private:
  ValType type;
  union {
    int intVal;
    float floatVal;
    bool boolVal;
    char stringVal[STRING_VAL_LEN + 1];
  } value;
};

#endif /* STATICVALUEREFERENCE_H_ */
