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

import polimi.trex.common.Consts.Op;
import polimi.trex.common.Consts.OpTreeType;
import polimi.trex.common.Consts.ValType;

/**
 * Represents an operation tree that is used to compute an operation over the values of detected events.
 */
public class OpTree {
	private OpTreeType type;
	private OpTree leftTree;
	private OpTree rightTree;
	private ValType valType;
	private OpValueReference valueRef;
	private Op op;

	/** Constructor used for inner nodes */
	public OpTree(OpTree leftTree, OpTree rightTree, Op op, ValType valType) {
		type = OpTreeType.INNER;
		this.leftTree = leftTree;
		this.rightTree = rightTree;
		this.op = op;
		this.valueRef = null;
		this.valType = valType;
	}
	
	/** Constructor used for leaf nodes */
	public OpTree(OpValueReference valueRef, ValType valType) {
		type = OpTreeType.LEAF;
		this.valueRef = valueRef;
		this.leftTree = null;
		this.rightTree = null;
		this.op = Op.ADD;
		this.valType = valType;
	}

	public OpTreeType getType() {
		return type;
	}

	public OpTree getLeftTree() {
		return leftTree;
	}

	public OpTree getRightTree() {
		return rightTree;
	}

	public OpValueReference getValueRef() {
		return valueRef;
	}

	public Op getOp() {
		return op;
	}
	
	public ValType getValType() {
		return valType;
	}
	
	public void setValType(ValType valType) {
		this.valType = valType;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (! (obj instanceof OpTree)) return false;
		OpTree other = (OpTree) obj;
		if (valType!=other.valType) return false;
		if (leftTree == null) {
			if (other.leftTree != null) return false;
		} else {
			if (! leftTree.equals(other.leftTree)) return false;
		}
		if (op != other.op) return false;
		if (rightTree == null) {
			if (other.rightTree != null) return false;
		} else {
			if (!rightTree.equals(other.rightTree)) return false;
		}
		if (type != other.type) return false;
		if (valueRef == null) {
			if (other.valueRef != null) {
				return false;
			}
		} else {
			if (!valueRef.equals(other.valueRef)) return false;
		}
		return true;
	}
}
