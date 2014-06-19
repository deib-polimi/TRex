package polimi.trex.common;

import polimi.trex.common.Consts.ValRefType;

public abstract class OpValueReference {
	private ValRefType refType;

	public ValRefType getRefType() {
		return refType;
	}

	public void setRefType(ValRefType refType) {
		this.refType = refType;
	}
}
