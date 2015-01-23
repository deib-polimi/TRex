//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Francesco Feltrinelli, Daniele Rogora
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

package polimi.trex.marshalling;

import java.util.Collection;
import java.util.Map;

import polimi.trex.common.Consts.EngineType;
import polimi.trex.common.TAggregate;
import polimi.trex.common.Attribute;
import polimi.trex.common.ComplexParameter;
import polimi.trex.common.Constraint;
import polimi.trex.common.Consts.ValRefType;
import polimi.trex.common.EventPredicate;
import polimi.trex.common.EventTemplate;
import polimi.trex.common.EventTemplateAttr;
import polimi.trex.common.EventTemplateStaticAttr;
import polimi.trex.common.Negation;
import polimi.trex.common.OpTree;
import polimi.trex.common.OpValueReference;
import polimi.trex.common.RulePktValueReference;
import polimi.trex.common.Consts.AggregateFun;
import polimi.trex.common.Consts.CompKind;
import polimi.trex.common.Consts.ConstraintOp;
import polimi.trex.common.Consts.Op;
import polimi.trex.common.Consts.OpTreeType;
import polimi.trex.common.Consts.StateType;
import polimi.trex.common.Consts.ValType;
import polimi.trex.common.StaticValueReference;
import polimi.trex.packets.AdvPkt;
import polimi.trex.packets.JoinPkt;
import polimi.trex.packets.PingPkt;
import polimi.trex.packets.PubPkt;
import polimi.trex.packets.RulePkt;
import polimi.trex.packets.SubPkt;
import polimi.trex.packets.TRexPkt;
import polimi.trex.packets.UnSubPkt;
import polimi.trex.packets.TRexPkt.PacketType;
import polimi.trex.packets.TRexPkt.PktType;


public class Marshaller {
	
	/**
	 * Number of bytes used to store the packet type.
	 */
	public final static int BYTENUM_PKTTYPE= 1;
	/**
	 * Number of bytes used to store the packet length.
	 */
	public final static int BYTENUM_PKTLENGTH= 4;
	
	public static byte[] getByteArray(PubPkt pkt) {
		byte[] dest = new byte[4+getNumBytes(PktType.PUB_PKT)+getNumBytes(pkt)];
		encode(pkt, dest, 0);
		return dest;
	}
	
	public static byte[] getByteArray(RulePkt pkt, EngineType type) {
		byte[] dest = new byte[4+getNumBytes(PktType.RULE_PKT)+getNumBytes(pkt)];
		encode(pkt, dest, 0, type);
		return dest;
	}
	
	public static byte[] getByteArray(SubPkt pkt) {
		byte[] dest = new byte[4+getNumBytes(PktType.SUB_PKT)+getNumBytes(pkt)];
		encode(pkt, dest, 0);
		return dest;
	}
	
	public static byte[] getByteArray(AdvPkt pkt) {
		byte[] dest = new byte[4+getNumBytes(PktType.ADV_PKT)+getNumBytes(pkt)];
		encode(pkt, dest, 0);
		return dest;
	}
	
	public static byte[] getByteArray(JoinPkt pkt) {
		byte[] dest = new byte[4+getNumBytes(PktType.JOIN_PKT)+getNumBytes(pkt)];
		encode(pkt, dest, 0);
		return dest;
	}
	
	protected static int getNumBytes(RulePkt pkt) {
		int size = 0;
		//size += getNumBytes(EngineType.CPU);
		size += getNumBytes(pkt.getPredicates());
		size += getNumBytes(pkt.getParameters());
		size += getNumBytes(pkt.getAggregates());
		size += getNumBytes(pkt.getNegations());
		size += getNumBytes(pkt.getEventTemplate());
		size += getNumBytes(pkt.getConsuming());
		return size;
	}
	
	protected static int getNumBytes(EngineType type) {
		return 1;
	}
	
	protected static int encode(EngineType type, byte[] dest, int startIndex) {
		if (type==EngineType.CPU) dest[startIndex++] = 0;
		else if (type==EngineType.GPU) dest[startIndex++] = 1;
		return startIndex;
	}
	
	protected static int encode(RulePkt source, byte[] dest, int startIndex, EngineType eType) {
		startIndex = encode(PktType.RULE_PKT, dest, startIndex);
		//startIndex = encode(eType, dest, startIndex);
		startIndex = encode(getNumBytes(source), dest, startIndex);
		startIndex = encode(source.getPredicates(), dest, startIndex);
		startIndex = encode(source.getParameters(), dest, startIndex);
		startIndex = encode(source.getAggregates(), dest, startIndex);
		startIndex = encode(source.getNegations(), dest, startIndex);
		startIndex = encode(source.getEventTemplate(), dest, startIndex);
		startIndex = encode(source.getConsuming(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(PubPkt pkt) {
		int size = 0;
		size += getNumBytes(pkt.getEventType());
		size += getNumBytes(pkt.getAttributes());
		return size;
	}
	
	protected static int encode(PubPkt source, byte[] dest, int startIndex) {
		startIndex = encode(PktType.PUB_PKT, dest, startIndex);
		startIndex = encode(getNumBytes(source), dest, startIndex);
		startIndex = encode(source.getEventType(), dest, startIndex);
		startIndex = encode(source.getAttributes(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(SubPkt pkt) {
		int size = 0;
		size += getNumBytes(pkt.getEventType());
		size += getNumBytes(pkt.getConstraints());
		return size;
	}
	
	protected static int encode(SubPkt source, byte[] dest, int startIndex) {
		startIndex = encode(PktType.SUB_PKT, dest, startIndex);
		startIndex = encode(getNumBytes(source), dest, startIndex);
		startIndex = encode(source.getEventType(), dest, startIndex);
		startIndex = encode(source.getConstraints(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(AdvPkt pkt) {
		int size = 0;
		size += getNumBytes(pkt.getAdvertisements());
		return size;
	}
	
	protected static int encode(AdvPkt source, byte[] dest, int startIndex) {
		startIndex = encode(PktType.ADV_PKT, dest, startIndex);
		startIndex = encode(getNumBytes(source), dest, startIndex);
		startIndex = encode(source.getAdvertisements(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(JoinPkt pkt) {
		int size = 0;
		size += getNumBytes(pkt.getAddress());
		size += getNumBytes(pkt.getPort());
		return size;
	}
	
	protected static int encode(JoinPkt source, byte[] dest, int startIndex) {
		startIndex = encode(PktType.JOIN_PKT, dest, startIndex);
		startIndex = encode(getNumBytes(source), dest, startIndex);
		startIndex = encode(source.getAddress(), dest, startIndex);
		startIndex = encode(source.getPort(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(Constraint constraint) {
		int size = 0;
		size += getNumBytes(constraint.getName());
		size += getNumBytes(constraint.getOp());
		size += getNumBytes(constraint.getValType());
		if (constraint.getValType()==ValType.INT) size += getNumBytes(constraint.getIntVal());
		else if (constraint.getValType()==ValType.FLOAT) size += getNumBytes(constraint.getFloatVal());
		else if (constraint.getValType()==ValType.BOOL) size += getNumBytes(constraint.getBoolVal());
		else if (constraint.getValType()==ValType.STRING) size += getNumBytes(constraint.getStringVal());
		return size;
	}
	
	protected static int encode(Constraint source, byte[] dest, int startIndex) {
		startIndex = encode(source.getName(), dest, startIndex);
		startIndex = encode(source.getOp(), dest, startIndex);
		startIndex = encode(source.getValType(), dest, startIndex);
		if (source.getValType()==ValType.INT) startIndex = encode(source.getIntVal(), dest, startIndex);
		else if (source.getValType()==ValType.FLOAT) startIndex = encode(source.getFloatVal(), dest, startIndex);
		else if (source.getValType()==ValType.BOOL) startIndex = encode(source.getBoolVal(), dest, startIndex);
		else if (source.getValType()==ValType.STRING) startIndex = encode(source.getStringVal(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(Attribute att) {
		int size = 0;
		size += getNumBytes(att.getName());
		size += getNumBytes(att.getValType());
		if (att.getValType()==ValType.INT) size += getNumBytes(att.getIntVal());
		else if (att.getValType()==ValType.FLOAT) size += getNumBytes(att.getFloatVal());
		else if (att.getValType()==ValType.BOOL) size += getNumBytes(att.getBoolVal());
		else if (att.getValType()==ValType.STRING) size += getNumBytes(att.getStringVal());
		return size;
	}

	protected static int encode(Attribute source, byte[] dest, int startIndex) {
		startIndex = encode(source.getName(), dest, startIndex);
		startIndex = encode(source.getValType(), dest, startIndex);
		if (source.getValType()==ValType.INT) startIndex = encode(source.getIntVal(), dest, startIndex);
		else if (source.getValType()==ValType.FLOAT) startIndex = encode(source.getFloatVal(), dest, startIndex);
		else if (source.getValType()==ValType.BOOL) startIndex = encode(source.getBoolVal(), dest, startIndex);
		else if (source.getValType()==ValType.STRING) startIndex = encode(source.getStringVal(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(EventPredicate pred) {
		int size = 0;
		size += getNumBytes(pred.getEventType());
		size += getNumBytes(pred.getConstraints());
		size += getNumBytes(pred.getRefersTo());
		size += getNumBytes(pred.getWin());
		size += getNumBytes(pred.getKind());
		return size;
	}

	protected static int encode(EventPredicate source, byte[] dest, int startIndex) {
		startIndex = encode(source.getEventType(), dest, startIndex);
		startIndex = encode(source.getConstraints(), dest, startIndex);
		startIndex = encode(source.getRefersTo(), dest, startIndex);
		startIndex = encode(source.getWin(), dest, startIndex);
		startIndex = encode(source.getKind(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(ComplexParameter par) {
		int size = 0;
		size += getNumBytes(par.getOperation());
		size += getNumBytes(par.getStateType());
		size += getNumBytes(par.getValueType());
		size += getNumBytes(par.getRightTree());
		size += getNumBytes(par.getLeftTree());
		return size;
	}

	protected static int encode(ComplexParameter source, byte[] dest, int startIndex) {
		startIndex = encode(source.getOperation(), dest, startIndex);
		startIndex = encode(source.getStateType(), dest, startIndex);
		startIndex = encode(source.getValueType(), dest, startIndex);
		startIndex = encode(source.getLeftTree(), dest, startIndex);
		startIndex = encode(source.getRightTree(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(Negation neg) {
		int size = 0;
		size += getNumBytes(neg.getEventType());
		size += getNumBytes(neg.getConstraints());
		size += getNumBytes(neg.getLowerId());
		size += getNumBytes(neg.getLowerTime());
		size += getNumBytes(neg.getUpperId());
		return size;
	}

	protected static int encode(Negation source, byte[] dest, int startIndex) {
		startIndex = encode(source.getEventType(), dest, startIndex);
		startIndex = encode(source.getConstraints(), dest, startIndex);
		startIndex = encode(source.getLowerId(), dest, startIndex);
		startIndex = encode(source.getLowerTime(), dest, startIndex);
		startIndex = encode(source.getUpperId(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(TAggregate agg) {
		int size = 0;
		size += getNumBytes(agg.getEventType());
		size += getNumBytes(agg.getConstraints());
		size += getNumBytes(agg.getLowerId());
		size += getNumBytes(agg.getLowerTime());
		size += getNumBytes(agg.getUpperId());
		size += getNumBytes(agg.getFun());
		size += getNumBytes(agg.getName());
		return size;
	}

	protected static int encode(TAggregate source, byte[] dest, int startIndex) {
		startIndex = encode(source.getEventType(), dest, startIndex);
		startIndex = encode(source.getConstraints(), dest, startIndex);
		startIndex = encode(source.getLowerId(), dest, startIndex);
		startIndex = encode(source.getLowerTime(), dest, startIndex);
		startIndex = encode(source.getUpperId(), dest, startIndex);
		startIndex = encode(source.getFun(), dest, startIndex);
		startIndex = encode(source.getName(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(EventTemplate template) {
		int size = 0;
		size += getNumBytes(template.getEventType());
		size += getNumBytes(template.getAttributes());
		size += getNumBytes(template.getStaticAttributes());
		return size;
	}

	protected static int encode(EventTemplate source, byte[] dest, int startIndex) {
		startIndex = encode(source.getEventType(), dest, startIndex);
		startIndex = encode(source.getAttributes(), dest, startIndex);
		startIndex = encode(source.getStaticAttributes(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(EventTemplateAttr templateAttr) {
		int size = 0;
		size += getNumBytes(templateAttr.getName());
		size += getNumBytes(templateAttr.getValue());
		return size;
	}
	
	protected static int getNumBytes(EventTemplateStaticAttr templateAttr) {
		int size = 0;
		size += getNumBytes(templateAttr.getAttr());
		return size;
	}

	protected static int encode(EventTemplateAttr source, byte[] dest, int startIndex) {
		startIndex = encode(source.getName(), dest, startIndex);
		startIndex = encode(source.getValue(), dest, startIndex);
		return startIndex;
	}
	
	protected static int encode(EventTemplateStaticAttr source, byte[] dest, int startIndex) {
		startIndex = encode(source.getAttr(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(OpTree tree) {
		int size = 0;
		size += getNumBytes(tree.getType());
		size += getNumBytes(tree.getValType());
		if (tree.getType()==OpTreeType.INNER) {
			size += getNumBytes(tree.getLeftTree());
			size += getNumBytes(tree.getRightTree());
			size += getNumBytes(tree.getOp());
		} else {
			size += getNumBytes(tree.getValueRef());
		}
		return size;
	}

	protected static int encode(OpTree source, byte[] dest, int startIndex) {
		startIndex = encode(source.getType(), dest, startIndex);
		startIndex = encode(source.getValType(), dest, startIndex);
		if (source.getType()==OpTreeType.INNER) {
			startIndex = encode(source.getLeftTree(), dest, startIndex);
			startIndex = encode(source.getRightTree(), dest, startIndex);
			startIndex = encode(source.getOp(), dest, startIndex);
		} else {
			startIndex = encode(source.getValueRef(), dest, startIndex);
		}
		return startIndex;
	}
	
	protected static int getNumBytes(ValRefType type) {
		return 1;
	}
	
	protected static int getNumBytes(OpValueReference valRef) {
		int size = 0;
		size += getNumBytes(valRef.getRefType());
		if (valRef instanceof RulePktValueReference) return getNumBytes((RulePktValueReference)valRef, size);
		else if (valRef instanceof StaticValueReference) return getNumBytes((StaticValueReference)valRef, size);
		return 0;
	}
	
	protected static int getNumBytes(RulePktValueReference valRef, int size) {
		size += getNumBytes(valRef.getIndex());
		size += getNumBytes(valRef.getStateType());
		size += getNumBytes(valRef.getName());
		return size;
	}
	
	protected static int getNumBytes(StaticValueReference valRef, int size) {
		size += getNumBytes(valRef.getType());
		if (valRef.getType() == ValType.INT) size += getNumBytes(valRef.getIntVal());
		if (valRef.getType() == ValType.FLOAT) size += getNumBytes(valRef.getFloatVal());
		if (valRef.getType() == ValType.BOOL) size += getNumBytes(valRef.getBoolVal());
		if (valRef.getType() == ValType.STRING) size += getNumBytes(valRef.getStringVal());
		return size;
	}
	
	protected static int encode(ValRefType type, byte[] dest, int startIndex) {
		if (type==ValRefType.RULEPKT) dest[startIndex++] = 0;
		else if (type==ValRefType.STATIC) dest[startIndex++] = 1;
		return startIndex;
	}

	protected static int encode(OpValueReference source, byte[] dest, int startIndex) {
		startIndex = encode(source.getRefType(), dest, startIndex);
		if (source instanceof RulePktValueReference) return encode((RulePktValueReference)source, dest, startIndex);
		else if (source instanceof StaticValueReference) return encode((StaticValueReference)source, dest, startIndex);
		return 0;
	}
	
	protected static int encode(RulePktValueReference source, byte[] dest, int startIndex) {
		startIndex = encode(source.getIndex(), dest, startIndex);
		startIndex = encode(source.getStateType(), dest, startIndex);
		startIndex = encode(source.getName(), dest, startIndex);
		return startIndex;
	}
	
	protected static int encode(StaticValueReference source, byte[] dest, int startIndex) {
		startIndex = encode(source.getType(), dest, startIndex);
		if (source.getType() == ValType.INT) startIndex = encode(source.getIntVal(), dest, startIndex);
		if (source.getType() == ValType.FLOAT) startIndex = encode(source.getFloatVal(), dest, startIndex);
		if (source.getType() == ValType.BOOL) startIndex = encode(source.getBoolVal(), dest, startIndex);
		if (source.getType() == ValType.STRING) startIndex = encode(source.getStringVal(), dest, startIndex);
		return startIndex;
	}
	
	protected static int getNumBytes(boolean val) {
		return 1;
	}
	
	protected static int encode(boolean source, byte[] dest, int startIndex) {
		dest[startIndex++] = (byte) (source ? 1 : 0);
		return startIndex;
	}
	
	protected static int getNumBytes(int num) {
		return 4;
	}
	
	protected static int encode(int source, byte[] dest, int startIndex) {
		dest[startIndex++] = (byte)((source >> 24) & 0xff);
		dest[startIndex++] = (byte)((source >> 16) & 0xff);
		dest[startIndex++] = (byte)((source >> 8) & 0xff);
		dest[startIndex++] = (byte)((source) & 0xff);
		return startIndex;
	}
	
	protected static int getNumBytes(float num) {
		return 4;
	}
	
	protected static int encode(float source, byte[] dest, int startIndex) {
		int i = Float.floatToRawIntBits(source);
		return encode(i, dest, startIndex);
	}
	
	protected static int getNumBytes(long num) {
		return 8;
	}
	
	protected static int encode(long source, byte[] dest, int startIndex) {
		dest[startIndex++] = (byte)((source >> 56) & 0xff);
		dest[startIndex++] = (byte)((source >> 48) & 0xff);
		dest[startIndex++] = (byte)((source >> 40) & 0xff);
		dest[startIndex++] = (byte)((source >> 32) & 0xff);
		dest[startIndex++] = (byte)((source >> 24) & 0xff);
		dest[startIndex++] = (byte)((source >> 16) & 0xff);
		dest[startIndex++] = (byte)((source >> 8) & 0xff);
		dest[startIndex++] = (byte)((source) & 0xff);
		return startIndex;
	}
	
	protected static int getNumBytes(String s) {
		return s.length()+4; // One int (4 bytes) more to encode the length of the string
	}

	protected static int encode(String source, byte[] dest, int startIndex) {
		int length = source.length();
		startIndex = encode(length, dest, startIndex);
		for (int i=0; i<length; i++) {
			dest[startIndex+i] = (byte) source.charAt(i);
		}
		return startIndex+length;
	}
	
	protected static int getNumBytes(CompKind kind) {
		return 1;
	}
	
	protected static int encode(CompKind source, byte[] dest, int startIndex) {
		if (source==CompKind.EACH_WITHIN) dest[startIndex++] = 0;
		else if (source==CompKind.FIRST_WITHIN) dest[startIndex++] = 1;
		else if (source==CompKind.LAST_WITHIN) dest[startIndex++] = 2;
		else if (source==CompKind.ALL_WITHIN) dest[startIndex++] = 3;
		return startIndex;
	}
	
	protected static int getNumBytes(ConstraintOp op) {
		return 1;
	}
	
	protected static int encode(ConstraintOp source, byte[] dest, int startIndex) {
		if (source==ConstraintOp.EQ) dest[startIndex++] = 0;
		else if (source==ConstraintOp.LT) dest[startIndex++] = 1;
		else if (source==ConstraintOp.GT) dest[startIndex++] = 2;
		else if (source==ConstraintOp.DF) dest[startIndex++] = 3;
		else if (source==ConstraintOp.IN) dest[startIndex++] = 4;
		else if (source==ConstraintOp.LE) dest[startIndex++] = 5;
		else if (source==ConstraintOp.GE) dest[startIndex++] = 6;
		return startIndex;
	}
	
	protected static int getNumBytes(ValType type) {
		return 1;
	}
	
	protected static int encode(ValType source, byte[] dest, int startIndex) {
		if (source==ValType.INT) dest[startIndex++] = 0;
		else if (source==ValType.FLOAT) dest[startIndex++] = 1;
		else if (source==ValType.BOOL) dest[startIndex++] = 2;
		else if (source==ValType.STRING) dest[startIndex++] = 3;
		return startIndex;
	}
	
	protected static int getNumBytes(StateType type) {
		return 1;
	}
	
	protected static int encode(StateType source, byte[] dest, int startIndex) {
		if (source==StateType.STATE) dest[startIndex++] = 0;
		else if (source==StateType.NEG) dest[startIndex++] = 1;
		else if (source==StateType.AGG) dest[startIndex++] = 2;
		return startIndex;
	}
	
	protected static int getNumBytes(AggregateFun fun) {
		return 1;
	}
	
	protected static int encode(AggregateFun source, byte[] dest, int startIndex) {
		if (source==AggregateFun.NONE) dest[startIndex++] = 0;
		else if (source==AggregateFun.AVG) dest[startIndex++] = 1;
		else if (source==AggregateFun.COUNT) dest[startIndex++] = 2;
		else if (source==AggregateFun.MIN) dest[startIndex++] = 3;
		else if (source==AggregateFun.MAX) dest[startIndex++] = 4;
		else if (source==AggregateFun.SUM) dest[startIndex++] = 5;
		return startIndex;
	}
	
	protected static int getNumBytes(OpTreeType type) {
		return 1;
	}
	
	protected static int encode(OpTreeType source, byte[] dest, int startIndex) {
		if (source==OpTreeType.LEAF) dest[startIndex++] = 0;
		else if (source==OpTreeType.INNER) dest[startIndex++] = 1;
		return startIndex;
	}
	
	protected static int getNumBytes(Op op) {
		return 1;
	}
	
	protected static int encode(Op source, byte[] dest, int startIndex) {
		if (source==Op.ADD) dest[startIndex++] = 0;
		else if (source==Op.SUB) dest[startIndex++] = 1;
		else if (source==Op.MUL) dest[startIndex++] = 2;
		else if (source==Op.DIV) dest[startIndex++] = 3;
		else if (source==Op.AND) dest[startIndex++] = 4;
		else if (source==Op.OR) dest[startIndex++] = 5;
		return startIndex;
	}
	
	protected static int getNumBytes(PktType type) {
		return 1;
	}
	
	protected static int encode(PktType source, byte[] dest, int startIndex) {
		if (source==PktType.PUB_PKT) dest[startIndex++] = 0;
		else if (source==PktType.RULE_PKT) dest[startIndex++] = 1;
		else if (source==PktType.SUB_PKT) dest[startIndex++] = 2;
		else if (source==PktType.ADV_PKT) dest[startIndex++] = 3;
		else if (source==PktType.JOIN_PKT) dest[startIndex++] = 4;
		return startIndex;
	}

	protected static int getNumBytes(Collection<? extends Object> collection) {
		int size = 4;	// One int (4 bytes) to encode the length of the collection
		for (Object el : collection) {
			if (el instanceof Integer) size += getNumBytes((Integer) el);
			else if (el instanceof Constraint) size += getNumBytes((Constraint) el);
			else if (el instanceof Attribute) size += getNumBytes((Attribute) el);
			else if (el instanceof EventTemplateAttr) size += getNumBytes((EventTemplateAttr) el);
			else if (el instanceof EventTemplateStaticAttr) size += getNumBytes((EventTemplateStaticAttr) el);
		}
		return size;
	}

	protected static int encode(Collection<? extends Object> source, byte[] dest, int startIndex) {
		int size = source.size();
		startIndex = encode(size, dest, startIndex);
		for (Object el : source) {
			if (el instanceof Integer) startIndex = encode((Integer) el, dest, startIndex);
			else if (el instanceof Constraint) startIndex = encode((Constraint) el, dest, startIndex);
			else if (el instanceof Attribute) startIndex = encode((Attribute) el, dest, startIndex);
			else if (el instanceof EventTemplateAttr) startIndex = encode((EventTemplateAttr) el, dest, startIndex);
			else if (el instanceof EventTemplateStaticAttr) startIndex = encode((EventTemplateStaticAttr) el, dest, startIndex);
		}
		return startIndex;
	}
	
	protected static int getNumBytes(Map<Integer, ? extends Object> map) {
		int size = 4;	// One int (4 bytes) to encode the length of the collection
		for (Integer key : map.keySet()) {
			size += getNumBytes(key);
			Object value = map.get(key);
			if (value instanceof EventPredicate) size += getNumBytes((EventPredicate) value);
			else if (value instanceof ComplexParameter) size += getNumBytes((ComplexParameter) value);
			else if (value instanceof Negation) size += getNumBytes((Negation) value);
			else if (value instanceof TAggregate) size += getNumBytes((TAggregate) value);
		}
		return size;
	}

	protected static int encode(Map<Integer, ? extends Object> source, byte[] dest, int startIndex) {
		int size = source.size();
		startIndex = encode(size, dest, startIndex);
		for (Integer key : source.keySet()) {
			startIndex = encode(key, dest, startIndex);
			Object value = source.get(key);
			if (value instanceof EventPredicate) startIndex = encode((EventPredicate) value, dest, startIndex);
			else if (value instanceof ComplexParameter) startIndex = encode((ComplexParameter) value, dest, startIndex);
			else if (value instanceof Negation) startIndex = encode((Negation) value, dest, startIndex);
			else if (value instanceof TAggregate) startIndex = encode((TAggregate) value, dest, startIndex);
		}
		return startIndex;
	}
	
	public static byte[] getByteArray(UnSubPkt pkt) {
		int bodyLen= getNumBytes(pkt);
		byte[] dest = new byte[BYTENUM_PKTTYPE + BYTENUM_PKTLENGTH + bodyLen];
		
		int startIndex= 0;
		startIndex= encode(PacketType.UNSUB_PACKET, dest, startIndex);
		startIndex= encode(bodyLen, dest, startIndex);
		startIndex= encode(pkt.getSubPkt(), dest, startIndex);
		
		return dest;
	}
	
	public static byte[] getByteArray(PingPkt pkt) {
		int bodyLen= getNumBytes(pkt);
		byte[] dest = new byte[BYTENUM_PKTTYPE + BYTENUM_PKTLENGTH + bodyLen];
		
		int startIndex= 0;
		startIndex= encode(PacketType.PING_PACKET, dest, startIndex);
		startIndex= encode(bodyLen, dest, startIndex);
		
		return dest;
	}
	
	protected static int getSize(SubPkt subPkt){
		return BYTENUM_PKTTYPE + BYTENUM_PKTLENGTH + Marshaller.getNumBytes(subPkt);
	}
	
	protected static int getNumBytes(UnSubPkt pkt) {
		return getSize(pkt.getSubPkt());
	}
	
	protected static int getNumBytes(PingPkt pkt) {
		return 0;
	}

	protected static int encode(PacketType source, byte[] dest, int startIndex) {
		dest[startIndex++]= (byte) source.toValue(); 
		return startIndex;
	}

/**
 * Flattens the given packet to an array of bytes.
 */
public static byte[] marshalRule(TRexPkt pkt, EngineType eType) {
	if (pkt instanceof RulePkt) return getByteArray((RulePkt) pkt, eType);
	return null;
}

/**
 * Flattens the given packet to an array of bytes.
 */
public static byte[] marshal(TRexPkt pkt) {
	if (pkt instanceof PubPkt) return getByteArray((PubPkt) pkt);
	else if (pkt instanceof SubPkt) return getByteArray((SubPkt) pkt);
	else if (pkt instanceof UnSubPkt) return getByteArray((UnSubPkt) pkt);
	else return getByteArray((PingPkt) pkt);
}
}
