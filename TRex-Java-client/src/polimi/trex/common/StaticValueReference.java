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

package polimi.trex.common;

import polimi.trex.common.Consts.ValRefType;
import polimi.trex.common.Consts.ValType;

public class StaticValueReference extends OpValueReference {
	private int intVal;
	private float floatVal;
	private String stringVal;
	private Boolean boolVal;
	private ValType type;
	
	public StaticValueReference(int pValue) {
		this.intVal = pValue;
		this.type = ValType.INT;
		this.setRefType(ValRefType.STATIC);
	}
	
	public StaticValueReference(float pValue) {
		this.floatVal = pValue;
		this.type = ValType.FLOAT;
		this.setRefType(ValRefType.STATIC);
	}
	
	public StaticValueReference(Boolean pValue) {
		this.boolVal = pValue;
		this.type = ValType.BOOL;
		this.setRefType(ValRefType.STATIC);
	}
	
	public StaticValueReference(String pValue) {
		this.stringVal = pValue;
		this.type = ValType.STRING;
		this.setRefType(ValRefType.STATIC);
	}
	
	public int getIntVal() {
		return intVal;
	}
	public void setIntVal(int intVal) {
		this.intVal = intVal;
	}
	public float getFloatVal() {
		return floatVal;
	}
	public void setFloatVal(float floatVal) {
		this.floatVal = floatVal;
	}
	public String getStringVal() {
		return stringVal;
	}
	public void setStringVal(String stringVal) {
		this.stringVal = stringVal;
	}
	public Boolean getBoolVal() {
		return boolVal;
	}
	public void setBoolVal(Boolean boolVal) {
		this.boolVal = boolVal;
	}
	public ValType getType() {
		return type;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof StaticValueReference)) return false;
		StaticValueReference other = (StaticValueReference) obj;
		if (type != other.type) return false;
		if (type==ValType.INT && intVal != other.getIntVal()) return false;
		if (type==ValType.FLOAT && floatVal != other.getFloatVal()) return false;
		if (type==ValType.BOOL && boolVal != other.getBoolVal()) return false;
		if (type==ValType.STRING && stringVal != other.getStringVal()) return false;
		return true;
	}

}
