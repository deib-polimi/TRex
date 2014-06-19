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
