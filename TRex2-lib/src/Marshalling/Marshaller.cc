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

#pragma GCC push_options
#pragma GCC optimize ("O0")

#include "Marshaller.h"

using namespace std;

char * Marshaller::encode(PubPkt *pkt) {
	char *array = new char[getNumBytes(pkt)+5]; // Pkt + Type + Size
	int startIndex = 0;
	startIndex = encode(pkt, array, startIndex);
	return array;
}

char * Marshaller::encode(RulePkt *pkt) {
	char *array = new char[getNumBytes(pkt)+5]; // Pkt + Type + Size
	int startIndex = 0;
	startIndex = encode(pkt, array, startIndex);
	return array;
}

char * Marshaller::encode(SubPkt *pkt) {
	char *array = new char[getNumBytes(pkt)+5];
	int startIndex = 0;
	startIndex = encode(pkt, array, startIndex);
	return array;
}

char * Marshaller::encode(AdvPkt *pkt) {
	char *array = new char[getNumBytes(pkt)+5];
	int startIndex = 0;
	startIndex = encode(pkt, array, startIndex);
	return array;
}

char * Marshaller::encode(JoinPkt *pkt) {
	char *array = new char[getNumBytes(pkt)+5];
	int startIndex = 0;
	startIndex = encode(pkt, array, startIndex);
	return array;
}

int Marshaller::getSize(PubPkt *pkt) {
	return getNumBytes(pkt) + 5;
}

int Marshaller::getSize(RulePkt *pkt) {
	return getNumBytes(pkt) + 5;
}

int Marshaller::getSize(SubPkt *pkt) {
	return getNumBytes(pkt) + 5;
}

int Marshaller::getSize(AdvPkt *pkt) {
	return getNumBytes(pkt) + 5;
}

int Marshaller::getSize(JoinPkt *pkt) {
	return getNumBytes(pkt) + 5;
}

int Marshaller::getNumBytes(RulePkt *pkt) {
	int size = 0;
	int numPredicates = pkt->getPredicatesNum();
	size += getNumBytes(numPredicates);
	for (int i=0; i<numPredicates; i++) {
		size += getNumBytes(i);
		size += getNumBytes(pkt->getPredicate(i));
	}
	int numParameters = pkt->getParametersNum();
	size += getNumBytes(numParameters);
	for (int i=0; i<numParameters; i++) {
		size += getNumBytes(i);
		size += getNumBytes(pkt->getParameter(i));
	}
	int numAggregates = pkt->getAggregatesNum();
	size += getNumBytes(numAggregates);
	for (int i=0; i<numAggregates; i++) {
		size += getNumBytes(i);
		size += getNumBytes(pkt->getAggregate(i));
	}
	int numNegations = pkt->getNegationsNum();
	size += getNumBytes(numNegations);
	for (int i=0; i<numNegations; i++) {
		size += getNumBytes(i);
		size += getNumBytes(pkt->getNegation(i));
	}
	size += getNumBytes(*(pkt->getCompositeEventTemplate()));
	set<int> consumingSet = pkt->getConsuming();
	int numConsuming = consumingSet.size();
	size += getNumBytes(numConsuming);
	for (set<int>::iterator it=consumingSet.begin(); it!=consumingSet.end(); ++it) {
		int consuming = *it;
		size += getNumBytes(consuming);
	}
	return size;
}

int Marshaller::encode(RulePkt *source, char *dest, int startIndex) {
	startIndex = encode(RULE_PKT, dest, startIndex);
	startIndex = encode(getNumBytes(source), dest, startIndex);
	int numPredicates = source->getPredicatesNum();
	startIndex = encode(numPredicates, dest, startIndex);
	for (int i=0; i<numPredicates; i++) {
		startIndex = encode(i, dest, startIndex);
		Predicate pred = source->getPredicate(i);
		startIndex = encode(pred, dest, startIndex);
	}
	int numParameters = source->getParametersNum();
	startIndex = encode(numParameters, dest, startIndex);
	for (int i=0; i<numParameters; i++) {
		startIndex = encode(i, dest, startIndex);
		Parameter par = source->getParameter(i);
		startIndex = encode(par, dest, startIndex);
	}
	int numAggregates = source->getAggregatesNum();
	startIndex = encode(numAggregates, dest, startIndex);
	for (int i=0; i<numAggregates; i++) {
		startIndex = encode(i, dest, startIndex);
		Aggregate agg = source->getAggregate(i);
		startIndex = encode(agg, dest, startIndex);
	}
	int numNegations = source->getNegationsNum();
	startIndex = encode(numNegations, dest, startIndex);
	for (int i=0; i<numNegations; i++) {
		startIndex = encode(i, dest, startIndex);
		Negation neg = source->getNegation(i);
		startIndex = encode(neg, dest, startIndex);
	}
	CompositeEventTemplate *compositeEventTemplate = source->getCompositeEventTemplate();
	startIndex = encode(*compositeEventTemplate, dest, startIndex);
	set<int> consuming = source->getConsuming();
	int consumingNum = consuming.size();
	startIndex = encode(consumingNum, dest, startIndex);
	for (set<int>::iterator it=consuming.begin(); it!=consuming.end(); ++it) {
		int consuming = *it;
		startIndex = encode(consuming, dest, startIndex);
	}
	return startIndex;
}

int Marshaller::getNumBytes(PubPkt *pkt) {
	int size = 0;
	size += getNumBytes(pkt->getEventType());
	size += getNumBytes(pkt->getTimeStamp());
	int attributesNum = pkt->getAttributesNum();
	size += getNumBytes(attributesNum);
	for (int i=0; i<attributesNum; i++) {
		Attribute att = pkt->getAttribute(i);
		size += getNumBytes(att);
	}
	return size;
}

int Marshaller::encode(PubPkt *source, char *dest, int startIndex) {
	startIndex = encode(PUB_PKT, dest, startIndex);
	startIndex = encode(getNumBytes(source), dest, startIndex);
	startIndex = encode(source->getEventType(), dest, startIndex);
	startIndex = encode(source->getTimeStamp(), dest, startIndex);
	int attributesNum = source->getAttributesNum();
	startIndex = encode(attributesNum, dest, startIndex);
	for (int i=0; i<attributesNum; i++) {
		Attribute att = source->getAttribute(i);
		startIndex = encode(att, dest, startIndex);
	}
	return startIndex;
}

int Marshaller::getNumBytes(SubPkt *pkt) {
	int size = 0;
	size += getNumBytes(pkt->getEventType());
	size += getNumBytes(pkt->getConstraintsNum());
	int constraintsNum = pkt->getConstraintsNum();
	size += getNumBytes(constraintsNum);
	for (int i=0; i<constraintsNum; i++) {
		Constraint constr = pkt->getConstraint(i);
		size += getNumBytes(constr);
	}
	return size;
}

int Marshaller::encode(SubPkt *source, char *dest, int startIndex) {
	startIndex = encode(SUB_PKT, dest, startIndex);
	startIndex = encode(getNumBytes(source), dest, startIndex);
	startIndex = encode(source->getEventType(), dest, startIndex);
	int constraintsNum = source->getConstraintsNum();
	startIndex = encode(constraintsNum, dest, startIndex);
	for (int i=0; i<constraintsNum; i++) {
		Constraint constr = source->getConstraint(i);
		startIndex = encode(constr, dest, startIndex);
	}
	return startIndex;
}

int Marshaller::getNumBytes(AdvPkt *pkt) {
	int size = 0;
	set<int> advertisements = pkt->getAdvertisements();
	size += getNumBytes(advertisements);
	return size;
}

int Marshaller::encode(AdvPkt *source, char *dest, int startIndex) {
	startIndex = encode(ADV_PKT, dest, startIndex);
	startIndex = encode(getNumBytes(source), dest, startIndex);
	set<int> advertisements = source->getAdvertisements();
	startIndex = encode(advertisements, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(JoinPkt *source) {
	int size = 0;
	size += source->getAddress();
	size += source->getPort();
	return size;
}

int Marshaller::encode(JoinPkt *source, char *dest, int startIndex) {
	startIndex = encode(JOIN_PKT, dest, startIndex);
	startIndex = encode(getNumBytes(source), dest, startIndex);
	startIndex = encode(source->getAddress(), dest, startIndex);
	startIndex = encode(source->getPort(), dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(Constraint &constraint) {
	int size = 0;
	size += getNumBytes(constraint.name);
	size += getNumBytes(constraint.op);
	size += getNumBytes(constraint.type);
	if (constraint.type==INT) size += getNumBytes(constraint.intVal);
	else if (constraint.type==FLOAT) size += getNumBytes(constraint.floatVal);
	else if (constraint.type==BOOL) size += getNumBytes(constraint.boolVal);
	else if (constraint.type==STRING) size += getNumBytes(constraint.stringVal);
	return size;
}

int Marshaller::encode(Constraint &source, char *dest, int startIndex) {
	startIndex = encode(source.name, dest, startIndex);
	startIndex = encode(source.op, dest, startIndex);
	startIndex = encode(source.type, dest, startIndex);
	if (source.type==INT) startIndex = encode(source.intVal, dest, startIndex);
	if (source.type==FLOAT) startIndex = encode(source.floatVal, dest, startIndex);
	if (source.type==BOOL) startIndex = encode(source.boolVal, dest, startIndex);
	if (source.type==STRING) startIndex = encode(source.stringVal, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(Attribute &source) {
	int size = 0;
	size += getNumBytes(source.name);
	size += getNumBytes(source.type);
	if (source.type==INT) size += getNumBytes(source.intVal);
	else if (source.type==FLOAT) size += getNumBytes(source.floatVal);
	else if (source.type==BOOL) size += getNumBytes(source.boolVal);
	else if (source.type==STRING) size += getNumBytes(source.stringVal);
	return size;
}

int Marshaller::encode(Attribute &source, char *dest, int startIndex) {
	startIndex = encode(source.name, dest, startIndex);
	startIndex = encode(source.type, dest, startIndex);
	if (source.type==INT) startIndex = encode(source.intVal, dest, startIndex);
	else if (source.type==FLOAT) startIndex = encode(source.floatVal, dest, startIndex);
	else if (source.type==BOOL) startIndex = encode(source.boolVal, dest, startIndex);
	else if (source.type==STRING) startIndex = encode(source.stringVal, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(Predicate &source) {
	int size = 0;
	size += getNumBytes(source.eventType);
	int constraintsNum = source.constraintsNum;
	size += getNumBytes(constraintsNum);
	for (int i=0; i<constraintsNum; i++) {
		size += getNumBytes(source.constraints[i]);
	}
	size += getNumBytes(source.refersTo);
	size += getNumBytes(source.win);
	size += getNumBytes(source.kind);
	return size;
}

int Marshaller::encode(Predicate &source, char *dest, int startIndex) {
	startIndex = encode(source.eventType, dest, startIndex);
	int constraintsNum = source.constraintsNum;
	startIndex = encode(constraintsNum, dest, startIndex);
	for (int i=0; i<constraintsNum; i++) {
		startIndex = encode(source.constraints[i], dest, startIndex);
	}
	startIndex = encode(source.refersTo, dest, startIndex);
	startIndex = encode(source.win, dest, startIndex);
	startIndex = encode(source.kind, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(Parameter &par) {
	int size = 0;
	size += getNumBytes(par.evIndex1);
	size += getNumBytes(par.name1);
	size += getNumBytes(par.evIndex2);
	size += getNumBytes(par.name2);
	size += getNumBytes(par.type);
	return size;
}

int Marshaller::encode(Parameter &source, char *dest, int startIndex) {
	startIndex = encode(source.evIndex1, dest, startIndex);
	startIndex = encode(source.name1, dest, startIndex);
	startIndex = encode(source.evIndex2, dest, startIndex);
	startIndex = encode(source.name2, dest, startIndex);
	startIndex = encode(source.type, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(Negation &neg) {
	int size = 0;
	size += getNumBytes(neg.eventType);
	int constraintsNum = neg.constraintsNum;
	size += getNumBytes(constraintsNum);
	for (int i=0; i<constraintsNum; i++) {
		size += getNumBytes(neg.constraints[i]);
	}
	size += getNumBytes(neg.lowerId);
	size += getNumBytes(neg.lowerTime);
	size += getNumBytes(neg.upperId);
	return size;
}

int Marshaller::encode(Negation &source, char *dest, int startIndex) {
	startIndex = encode(source.eventType, dest, startIndex);
	int constraintsNum = source.constraintsNum;
	startIndex = encode(constraintsNum, dest, startIndex);
	for (int i=0; i<constraintsNum; i++) {
		startIndex = encode(source.constraints[i], dest, startIndex);
	}
	startIndex = encode(source.lowerId, dest, startIndex);
	startIndex = encode(source.lowerTime, dest, startIndex);
	startIndex = encode(source.upperId, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(Aggregate &source) {
	int size = 0;
	size += getNumBytes(source.eventType);
	int numConstraints = source.constraintsNum;
	size += getNumBytes(numConstraints);
	for (int i=0; i<numConstraints; i++) {
		size += getNumBytes(source.constraints[i]);
	}
	size += getNumBytes(source.lowerId);
	size += getNumBytes(source.lowerTime);
	size += getNumBytes(source.upperId);
	size += getNumBytes(source.fun);
	size += getNumBytes(source.name);
	return size;
}

int Marshaller::encode(Aggregate &source, char *dest, int startIndex) {
	startIndex = encode(source.eventType, dest, startIndex);
	int numConstraints = source.constraintsNum;
	startIndex = encode(numConstraints, dest, startIndex);
	for (int i=0; i<numConstraints; i++) {
		startIndex = encode(source.constraints[i], dest, startIndex);
	}
	startIndex = encode(source.lowerId, dest, startIndex);
	startIndex = encode(source.lowerTime, dest, startIndex);
	startIndex = encode(source.upperId, dest, startIndex);
	startIndex = encode(source.fun, dest, startIndex);
	startIndex = encode(source.name, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(CompositeEventTemplate &source) {
	int size = 0;
	size += getNumBytes(source.getEventType());
	int numAttributes = source.getAttributesNum();
	size += getNumBytes(numAttributes);
	for (int i=0; i<numAttributes; i++) {
		size += getNumBytes(source.getAttributeName(i));
		size += getNumBytes(source.getAttributeTree(i));
	}
	return size;
}

int Marshaller::encode(CompositeEventTemplate &source, char *dest, int startIndex) {
	startIndex = encode(source.getEventType(), dest, startIndex);
	int numAttributes = source.getAttributesNum();
	startIndex = encode(numAttributes, dest, startIndex);
	for (int i=0; i<numAttributes; i++) {
		startIndex = encode(source.getAttributeName(i), dest, startIndex);
		startIndex = encode(source.getAttributeTree(i), dest, startIndex);
	}
	return startIndex;
}

int Marshaller::getNumBytes(CompositeEventTemplateAttr &source) {
	int size = 0;
	size += getNumBytes(source.getName());
	size += getNumBytes(source.getValue());
	return size;
}

int Marshaller::encode(CompositeEventTemplateAttr &source, char *dest, int startIndex) {
	startIndex = encode(source.getName(), dest, startIndex);
	startIndex = encode(source.getValue(), dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(OpTree *tree) {
	int size = 0;
	size += getNumBytes(tree->getType());
	size += getNumBytes(tree->getValType());
	if (tree->getType()==INNER) {
		size += getNumBytes(tree->getLeftSubtree());
		size += getNumBytes(tree->getRightSubtree());
		size += getNumBytes(tree->getOp());
	} else {
		size += getNumBytes((RulePktValueReference *) tree->getValueReference());
	}
	return size;
}

int Marshaller::encode(OpTree *source, char *dest, int startIndex) {
	startIndex = encode(source->getType(), dest, startIndex);
	startIndex = encode(source->getValType(), dest, startIndex);
	if (source->getType()==INNER) {
		startIndex = encode(source->getLeftSubtree(), dest, startIndex);
		startIndex = encode(source->getRightSubtree(), dest, startIndex);
		startIndex = encode(source->getOp(), dest, startIndex);
	} else {
		startIndex = encode((RulePktValueReference *) source->getValueReference(), dest, startIndex);
	}
	return startIndex;
}

int Marshaller::getNumBytes(RulePktValueReference *source) {
	int size = 0;
	size += getNumBytes(source->getIndex());
	size += getNumBytes(source->getSType());
	char *name = source->getName();
	if (name==NULL) {
		name = new char[2];
		name[0] = '0';
		name[1] = '\0';
	}
	size += getNumBytes(name);
	return size;
}

int Marshaller::encode(RulePktValueReference *source, char *dest, int startIndex) {
	startIndex = encode(source->getIndex(), dest, startIndex);
	startIndex = encode(source->getSType(), dest, startIndex);
	char *name = source->getName();
	if (name==NULL) {
		name = new char[2];
		name[0] = '0';
		name[1] = '\0';
	}
	startIndex = encode(name, dest, startIndex);
	return startIndex;
}

int Marshaller::getNumBytes(set<int> &source) {
	int numElements = source.size();
	int size = getNumBytes(numElements)*(numElements+1);
	return size;
}

int Marshaller::encode(set<int> &source, char *dest, int startIndex) {
	int numElements = source.size();
	startIndex = encode(numElements, dest, startIndex);
	for (set<int>::iterator it=source.begin(); it!=source.end(); ++it) {
		int element = *it;
		startIndex = encode(element, dest, startIndex);
	}
	return startIndex;
}

int Marshaller::getNumBytes(bool source) {
	return 1;
}

int Marshaller::encode(bool source, char *dest, int startIndex) {
	if (source) dest[startIndex++] = 1;
	else dest[startIndex++] = 0;
	return startIndex;
}

int Marshaller::getNumBytes(int source) {
	return 4;
}

int Marshaller::encode(int source, char *dest, int startIndex) {
	dest[startIndex] = ((source >> 24) & 0xff);
	dest[startIndex+1] = ((source >> 16) & 0xff);
	dest[startIndex+2] = ((source >> 8) & 0xff);
	dest[startIndex+3] = ((source >> 0) & 0xff);
	startIndex+=4;
	return startIndex;
}

int Marshaller::getNumBytes(uint64_t source) {
	return 8;
}

int Marshaller::encode(uint64_t source, char *dest, int startIndex) {
  dest[startIndex] = (char)((source >> 56) & 0xff);
	dest[startIndex+1] = (char)((source >> 48) & 0xff);
	dest[startIndex+2] = (char)((source >> 40) & 0xff);
	dest[startIndex+3] = (char)((source >> 32) & 0xff);
	dest[startIndex+4] = (char)((source >> 24) & 0xff);
	dest[startIndex+5] = (char)((source >> 16) & 0xff);
	dest[startIndex+6] = (char)((source >> 8) & 0xff);
	dest[startIndex+7] = (char)((source >> 0) & 0xff);
	startIndex+=8;
	return startIndex;
}

int Marshaller::getNumBytes(long source) {
	return 8;
}

int Marshaller::encode(float source, char *dest, int startIndex) {
	int * ptr = (int *) (& source) ;
	dest[startIndex] = ((*ptr >> 24) & 0xff);
	dest[startIndex+1] = ((*ptr >> 16) & 0xff);
	dest[startIndex+2] = ((*ptr >> 8) & 0xff);
	dest[startIndex+3] = ((*ptr >> 0) & 0xff);
	startIndex+=4;
	return startIndex;
}

int Marshaller::getNumBytes(float source) {
	return 4;
}

int Marshaller::encode(long source, char *dest, int startIndex) {
	dest[startIndex] = (char)((source >> 56) & 0xff);
	dest[startIndex+1] = (char)((source >> 48) & 0xff);
	dest[startIndex+2] = (char)((source >> 40) & 0xff);
	dest[startIndex+3] = (char)((source >> 32) & 0xff);
	dest[startIndex+4] = (char)((source >> 24) & 0xff);
	dest[startIndex+5] = (char)((source >> 16) & 0xff);
	dest[startIndex+6] = (char)((source >> 8) & 0xff);
	dest[startIndex+7] = (char)((source >> 0) & 0xff);
	startIndex+=8;
	return startIndex;
}

int Marshaller::getNumBytes(char *source) {
	return strlen(source)+4;
}

int Marshaller::encode(char *source, char *dest, int startIndex) {
	int strLen = strlen(source);
	startIndex = encode(strLen, dest, startIndex);
	for (int i=0; i<strLen; i++) dest[startIndex+i] = source[i];
	startIndex += strLen;
	return startIndex;
}

int Marshaller::getNumBytes(CompKind source) {
	return 1;
}

int Marshaller::encode(CompKind source, char *dest, int startIndex) {
	if (source==EACH_WITHIN) dest[startIndex++] = 0;
	else if (source==FIRST_WITHIN) dest[startIndex++] = 1;
	else if (source==LAST_WITHIN) dest[startIndex++] = 2;
	else if (source==ALL_WITHIN) dest[startIndex++] = 3;
	return startIndex;
}

int Marshaller::getNumBytes(Op source) {
	return 1;
}

int Marshaller::encode(Op source, char *dest, int startIndex) {
	if (source==EQ) dest[startIndex++] = 0;
	else if (source==LT) dest[startIndex++] = 1;
	else if (source==GT) dest[startIndex++] = 2;
	else if (source==DF) dest[startIndex++] = 3;
	else if (source==IN) dest[startIndex++] = 4;
	else if (source==LE) dest[startIndex++] = 5;
	else if (source==GE) dest[startIndex++] = 6;
	return startIndex;
}

int Marshaller::getNumBytes(StateType source) {
	return 1;
}

int Marshaller::encode(StateType source, char *dest, int startIndex) {
	if (source==STATE) dest[startIndex++] = 0;
	else if (source==NEG) dest[startIndex++] = 1;
	else if (source==AGG) dest[startIndex++] = 2;
	return startIndex;
}

int Marshaller::getNumBytes(ValType source) {
	return 1;
}

int Marshaller::encode(ValType source, char *dest, int startIndex) {
	if (source==INT) dest[startIndex++] = 0;
	else if (source==FLOAT) dest[startIndex++] = 1;
	else if (source==BOOL) dest[startIndex++] = 2;
	else if (source==STRING) dest[startIndex++] = 3;
	return startIndex;
}

int Marshaller::getNumBytes(AggregateFun source) {
	return 1;
}

int Marshaller::encode(AggregateFun source, char *dest, int startIndex) {
	if (source==NONE) dest[startIndex++] = 0;
	else if (source==AVG) dest[startIndex++] = 1;
	else if (source==COUNT) dest[startIndex++] = 2;
	else if (source==MIN) dest[startIndex++] = 3;
	else if (source==MAX) dest[startIndex++] = 4;
	else if (source==SUM) dest[startIndex++] = 5;
	return startIndex;
}

int Marshaller::getNumBytes(OpTreeType source) {
	return 1;
}

int Marshaller::encode(OpTreeType source, char *dest, int startIndex) {
	if (source==LEAF) dest[startIndex++] = 0;
	else if (source==INNER) dest[startIndex++] = 1;
	return startIndex;
}

int Marshaller::getNumBytes(OpTreeOperation source) {
	return 1;
}

int Marshaller::encode(OpTreeOperation source, char *dest, int startIndex) {
	if (source==ADD) dest[startIndex++] = 0;
	else if (source==SUB) dest[startIndex++] = 1;
	else if (source==MUL) dest[startIndex++] = 2;
	else if (source==DIV) dest[startIndex++] = 3;
	else if (source==AND) dest[startIndex++] = 4;
	else if (source==OR) dest[startIndex++] = 5;
	return startIndex;
}

int Marshaller::getNumBytes(PktType source) {
	return 1;
}

int Marshaller::encode(PktType source, char *dest, int startIndex) {
	if (source==PUB_PKT) dest[startIndex] = 0;
	else if (source==RULE_PKT) dest[startIndex] = 1;
	else if (source==SUB_PKT) dest[startIndex] = 2;
	else if (source==ADV_PKT) dest[startIndex] = 3;
	else if (source==JOIN_PKT) dest[startIndex] = 4;
	return startIndex+1;
}

int Marshaller::getNumBytes(TimeMs source) {
	return 8;
}

int Marshaller::encode(TimeMs source, char *dest, int startIndex) {
	startIndex = encode(source.getTimeVal(), dest, startIndex);
	return startIndex;
}

#pragma GCC pop_options
