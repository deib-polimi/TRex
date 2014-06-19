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

package polimi.trex.common;

import polimi.trex.common.Consts.ConstraintOp;
import polimi.trex.common.Consts.ValType;

/**
 * Define a constraint, i.e. a basic expression that has to be satisfied by an event attribute.
 * It is used inside a RulePkt to provide content based filtering. 
 */
public class Constraint {
	private String name;
	private ConstraintOp op;
	private ValType valType;
	private int intVal;
	private float floatVal;
	private boolean boolVal;
	private String stringVal;
	
	private Constraint() {
		intVal = 0;
		floatVal = 0;
		boolVal = false;
		stringVal = "";
	}
	
	public Constraint(String name, ConstraintOp op, int val) {
		this();
		this.name = name;
		this.op = op;
		this.intVal = val;
		this.valType = ValType.INT;
	}
	
	public Constraint(String name, ConstraintOp op, float val) {
		this();
		this.name = name;
		this.op = op;
		this.floatVal = val;
		this.valType = ValType.FLOAT;
	}
	
	public Constraint(String name, ConstraintOp op, boolean val) {
		this();
		this.name = name;
		this.op = op;
		this.boolVal = val;
		this.valType = ValType.BOOL;
	}
	
	public Constraint(String name, ConstraintOp op, String val) {
		this();
		this.name = name;
		this.op = op;
		this.stringVal = val;
		this.valType = ValType.STRING;
	}
	
	public ValType getValType() {
		return valType;
	}
	
	public void setValType(ValType valType) {
		this.valType = valType;
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

	public boolean getBoolVal() {
		return boolVal;
	}

	public void setBoolVal(boolean boolVal) {
		this.boolVal = boolVal;
	}

	public String getStringVal() {
		return stringVal;
	}

	public void setStringVal(String stringVal) {
		this.stringVal = stringVal;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public ConstraintOp getOp() {
		return op;
	}

	public void setOp(ConstraintOp op) {
		this.op = op;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof Constraint)) return false;
		Constraint other = (Constraint) obj;
		if (valType!=other.valType) return false;
		if (! name.equals(other.name)) return false;
		if (op!=other.op) return false;
		if (valType==ValType.INT) if (intVal != other.intVal) return false;
		else if (valType==ValType.FLOAT) if (floatVal != other.floatVal) return false;
		else if (valType==ValType.BOOL) if (boolVal != other.boolVal) return false;
		else if (valType==ValType.STRING) if (! stringVal.equals(other.stringVal)) return false;
		return true;
	}
}
