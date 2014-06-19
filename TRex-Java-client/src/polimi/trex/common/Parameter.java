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

import polimi.trex.common.Consts.StateType;

/**
 * Defines a parameter, i.e. an equality between the value of two attributes in a rule.
 * It is used inside a RulePkt to provide content-based filtering
 */
public class Parameter {
	private int evIndex1;
	private String name1;
	private int evIndex2;
	private String name2;
	private StateType type;
	
	public Parameter(int evIndex1, String name1, int evIndex2, String name2, StateType type) {
		this.evIndex1 = evIndex1;
		this.name1 = name1;
		this.evIndex2 = evIndex2;
		this.name2 = name2;
		this.type = type;
	}

	public int getEvIndex1() {
		return evIndex1;
	}

	public void setEvIndex1(int evIndex1) {
		this.evIndex1 = evIndex1;
	}

	public String getName1() {
		return name1;
	}

	public void setName1(String name1) {
		this.name1 = name1;
	}

	public int getEvIndex2() {
		return evIndex2;
	}

	public void setEvIndex2(int evIndex2) {
		this.evIndex2 = evIndex2;
	}

	public String getName2() {
		return name2;
	}

	public void setName2(String name2) {
		this.name2 = name2;
	}
	
	public StateType getType() {
		return type;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof Parameter)) return false;
		Parameter other = (Parameter) obj;
		if (evIndex1 != other.evIndex1) return false;
		if (evIndex2 != other.evIndex2) return false;
		if (! name1.equals(other.name1)) return false;
		if (! name2.equals(other.name2)) return false;
		if (type != other.type) return false;
		return true;
	}
}