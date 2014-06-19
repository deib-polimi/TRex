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

#include "StaticValueReference.h"

StaticValueReference::StaticValueReference(int pValue) {
      type = INT;
      value.intVal = pValue;
      vrtype = STATIC;
}

StaticValueReference::StaticValueReference(float pValue) {
      type = FLOAT;
      value.floatVal = pValue;
      vrtype = STATIC;
}

StaticValueReference::StaticValueReference(bool pValue) {
      type = BOOL;
      value.boolVal = pValue;
      vrtype = STATIC;
}

StaticValueReference::StaticValueReference(char *pValue) {
      type = STRING;
      strcpy(value.stringVal, pValue);
      vrtype = STATIC;
}

StaticValueReference::~StaticValueReference() {
	
}

ValType StaticValueReference::getType()
{
      return type;
}

int StaticValueReference::getIntValue()
{
      return value.intVal;
}

bool StaticValueReference::getBoolValue()
{
      return value.boolVal;
}

float StaticValueReference::getFloatValue()
{
      return value.floatVal;
}

void StaticValueReference::getStringValue(char* pValue)
{
      strcpy(pValue, value.stringVal);
}


OpValueReference * StaticValueReference::dup() {
      switch (type) {
	case INT:
	  return new StaticValueReference(value.intVal);
	  
	case FLOAT:
	  return new StaticValueReference(value.floatVal);
	  
	case BOOL:
	  return new StaticValueReference(value.boolVal);
	  
	case STRING:
	  return new StaticValueReference(value.stringVal);
      }
}
