package polimi.trex.common;

import polimi.trex.common.Consts.ValType;

public class EventTemplateStaticAttr {
	private Attribute attr;
	
	public EventTemplateStaticAttr(Attribute pAttr) {
		this.attr = pAttr;
	}
	
	public EventTemplateStaticAttr(String name, int value) {
		this.attr = new Attribute(name, value);
	}
	
	public EventTemplateStaticAttr(String name, float value) {
		this.attr = new Attribute(name, value);
	}
	
	public EventTemplateStaticAttr(String name, Boolean value) {
		this.attr = new Attribute(name, value);
	}
	
	public EventTemplateStaticAttr(String name, String value) {
		this.attr = new Attribute(name, value);
	}
	
	public Attribute getAttr() {
		return attr;
	}

	public String getName() {
		return attr.getName();
	}
	
	public ValType getValType() {
		return attr.getValType();
	}
	
	public int getIntVal() {
		return attr.getIntVal();
	}
	
	public float getFloatVal() {
		return attr.getFloatVal();
	}

	public Boolean getBoolVal() {
		return attr.getBoolVal();
	}

	public String getStringVal() {
		return attr.getStringVal();
	}

}
