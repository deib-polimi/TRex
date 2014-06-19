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
