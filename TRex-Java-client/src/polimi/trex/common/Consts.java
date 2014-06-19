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

public class Consts {
	/** Kinds of compositions */
	public enum CompKind {
		EACH_WITHIN,
		FIRST_WITHIN,
		LAST_WITHIN,
		ALL_WITHIN
	}

	/** Operations used in constraints */
	public enum ConstraintOp {
		EQ,
		LT,
		GT,
		DF,
		IN
	}

	/** Aggregate functions defined */
	public enum AggregateFun {
		NONE,
		AVG,
		COUNT,
		MIN,
		MAX,
		SUM,
	}
	
	public enum EngineType {
		CPU,
		GPU;
	}

	/** Type of the state */
	public enum StateType {
		STATE,
		NEG,
		AGG
	}
	
	/** Type of OpTree */
	public enum OpTreeType {
		LEAF,
		INNER;
	}
	
	/** Type of OpTree */
	public enum ValRefType {
		RULEPKT,
		STATIC;
	}
	
	/** Type of the value in an attribute or constraint */
	public enum ValType {
		INT,
		FLOAT,
		BOOL,
		STRING
	}
	
	/** Operations used to compute the values for complex events' attributes */
	public enum Op {
		ADD,
		SUB,
		MUL,
		DIV,
		AND,
		OR
	}
}
