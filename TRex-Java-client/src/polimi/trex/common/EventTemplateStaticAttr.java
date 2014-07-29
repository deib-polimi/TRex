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
