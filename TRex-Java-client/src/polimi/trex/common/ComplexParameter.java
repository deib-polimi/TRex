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

import polimi.trex.common.Consts.ConstraintOp;
import polimi.trex.common.Consts.StateType;
import polimi.trex.common.Consts.ValType;

public class ComplexParameter {
	private ValType type;
	private OpTree leftTree;
	private OpTree rightTree;
	private ConstraintOp operation;
	private StateType sType;

	
	public ComplexParameter(ConstraintOp op, StateType pSType, ValType vType, OpTree pLeftTree, OpTree pRightTree) {
		this.operation 	= op;
		this.sType		= pSType;
		this.type		= vType;
		this.leftTree	= pLeftTree;
		this.rightTree	= pRightTree;
	}
	
	public ValType getValueType() {
		return type;
	}
	
	public StateType getStateType() {
		return sType;
	}

	public OpTree getLeftTree() {
		return leftTree;
	}

	public OpTree getRightTree() {
		return rightTree;
	}

	public ConstraintOp getOperation() {
		return operation;
	}

}
