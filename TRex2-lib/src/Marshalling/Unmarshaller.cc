//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Daniele Rogora
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

#include "Unmarshaller.h"
#include <assert.h>

using namespace std;

RulePkt * Unmarshaller::decodeRulePkt(char *source) {
	int index = 0;
	return decodeRulePkt(source, index);
}

PubPkt * Unmarshaller::decodePubPkt(char *source) {
	int index = 0;
	return decodePubPkt(source, index);
}

SubPkt * Unmarshaller::decodeSubPkt(char *source) {
	int index = 0;
	return decodeSubPkt(source, index);
}

AdvPkt * Unmarshaller::decodeAdvPkt(char *source) {
	int index = 0;
	return decodeAdvPkt(source, index);
}

JoinPkt * Unmarshaller::decodeJoinPkt(char *source) {
	int index = 0;
	return decodeJoinPkt(source, index);
}

int Unmarshaller::decodeInt(char *source) {
	int index = 0;
	return decodeInt(source, index);
}

PktType Unmarshaller::decodePktType(char *source) {
	int index = 0;
	return decodePktType(source, index);
}

RulePkt * Unmarshaller::decodeRulePkt(char *source, int &index) {
  	int predicatesNum = decodeInt(source, index);
	map<int, Predicate> predicates;
	for (int i=0; i<predicatesNum; i++) {
		int key = decodeInt(source, index);
		Predicate value = decodeEventPredicate(source, index);
		predicates.insert(make_pair(key, value));
	}
	int parametersNum = decodeInt(source, index);
	map<int, ComplexParameter> parameters;
	for (int i=0; i<parametersNum; i++) {
		int key = decodeInt(source, index);
		ComplexParameter value = decodeParameter(source, index);
		parameters.insert(make_pair(key, value));
	}
	int aggregatesNum = decodeInt(source, index);
	map<int, Aggregate> aggregates;
	for (int i=0; i<aggregatesNum; i++) {
		int key = decodeInt(source, index);
		Aggregate value = decodeAggregate(source, index);
		aggregates.insert(make_pair(key, value));
	}
	int negationsNum = decodeInt(source, index);
	map<int, Negation> negations;
	for (int i=0; i<negationsNum; i++) {
		int key = decodeInt(source, index);
		Negation value = decodeNegation(source, index);
		negations.insert(make_pair(key, value));
	}
	CompositeEventTemplate *eventTemplate = decodeEventTemplate(source, index);
	int numConsuming = decodeInt(source, index);
	set<int> consuming;
	for (int i=0; i<numConsuming; i++) {
		consuming.insert(decodeInt(source, index));
	}
	RulePkt *pkt = new RulePkt(false);
	for (map<int, Predicate>::iterator it=predicates.begin(); it!=predicates.end(); ++it) {
		int index = it->first;
		Predicate pred = it->second;
		
		if (index==0) {
			pkt->addRootPredicate(pred.eventType, pred.constraints, pred.constraintsNum);
		} else {
			pkt->addPredicate(pred.eventType, pred.constraints, pred.constraintsNum, pred.refersTo, pred.win, pred.kind);
		}
		delete pred.constraints;
	}
	for (map<int, Aggregate>::iterator it=aggregates.begin(); it!=aggregates.end(); ++it) {
		Aggregate agg = it->second;
		if (agg.lowerId<0) {
			pkt->addTimeBasedAggregate(agg.eventType, agg.constraints, agg.constraintsNum, agg.upperId, agg.lowerTime, agg.name, agg.fun);
		} else {
			pkt->addAggregateBetweenStates(agg.eventType, agg.constraints, agg.constraintsNum, agg.lowerId, agg.upperId, agg.name, agg.fun);
		}
		delete agg.constraints;
	}
	for (map<int, Negation>::iterator it=negations.begin(); it!=negations.end(); ++it) {
		Negation neg = it->second;
		if (neg.lowerId<0) {
			pkt->addTimeBasedNegation(neg.eventType, neg.constraints, neg.constraintsNum, neg.upperId, neg.lowerTime);
		} else {
			pkt->addNegationBetweenStates(neg.eventType, neg.constraints, neg.constraintsNum, neg.lowerId, neg.upperId);
		}
		delete neg.constraints;
	}
	for (map<int, ComplexParameter>::iterator it=parameters.begin(); it!=parameters.end(); ++it) {
		ComplexParameter par = it->second;
		if (par.type==STATE) {
			pkt->addComplexParameter(par.operation, par.vtype, par.leftTree, par.rightTree);
		} else if (par.type==AGG) {
			pkt->addComplexParameterForAggregate(par.operation, par.vtype, par.leftTree, par.rightTree);
		} else {
			pkt->addComplexParameterForNegation(par.operation, par.vtype, par.leftTree, par.rightTree);
		}
	}
	pkt->setCompositeEventTemplate(eventTemplate);
	for (set<int>::iterator it=consuming.begin(); it!=consuming.end(); ++it) {
		int consum = *it;
		pkt->addConsuming(consum);
	}
	return pkt;
}

PubPkt * Unmarshaller::decodePubPkt(char *source, int &index) {
	int eventType = decodeInt(source, index);
	TimeMs timeStamp = decodeLong(source, index);
	int numAttributes = decodeInt(source, index);
	Attribute attributes[numAttributes];
	for (int i=0; i<numAttributes; i++) {
		attributes[i] = decodeAttribute(source, index);
	}
	PubPkt *pkt = new PubPkt(eventType, attributes, numAttributes);
	pkt->setTime(timeStamp);
	return pkt;
}

SubPkt * Unmarshaller::decodeSubPkt(char *source, int &index) {
	int eventType = decodeInt(source, index);
	int numConstraints = decodeInt(source, index);
	if (numConstraints==0) return new SubPkt(eventType);
	Constraint constraints[numConstraints];
	for (int i=0; i<numConstraints; i++) {
		constraints[i] = decodeConstraint(source, index);
	}
	SubPkt *pkt = new SubPkt(eventType, constraints, numConstraints);
	return pkt;
}

AdvPkt * Unmarshaller::decodeAdvPkt(char *source, int &index) {
	set<int> advertisements = decodeIntSet(source, index);
	return new AdvPkt(advertisements);
}

JoinPkt * Unmarshaller::decodeJoinPkt(char *source, int &index) {
	long address = decodeLong(source, index);
	int port = decodeInt(source, index);
	return new JoinPkt(address, port);
}

Constraint Unmarshaller::decodeConstraint(char *source, int &index) {
	Constraint c;
	char *name = decodeString(source, index);
	strcpy(c.name, name);
	delete name;
	c.op = decodeConstraintOp(source, index);
	c.type = decodeValType(source, index);
	if (c.type==INT) c.intVal = decodeInt(source, index);
	else if (c.type==FLOAT) c.floatVal = decodeFloat(source, index);
	else if (c.type==BOOL) c.boolVal = decodeBoolean(source, index);
	else if (c.type==STRING) {
		char *stringVal = decodeString(source, index);
		strcpy(c.stringVal, stringVal);
		delete stringVal;
	}
	return c;
}

Attribute Unmarshaller::decodeAttribute(char *source, int &index) {
	Attribute att;
	char *name = decodeString(source, index);
	assert(strlen(name) < NAME_LEN);
	strcpy(att.name, name);
	delete name;
	att.type = decodeValType(source, index);
	if (att.type==INT) att.intVal = decodeInt(source, index);
	else if (att.type==FLOAT) att.floatVal = decodeFloat(source, index);
	else if (att.type==BOOL) att.boolVal = decodeBoolean(source, index);
	else if (att.type==STRING) {
		char *stringVal = decodeString(source, index);
		assert(strlen(stringVal) < STRING_VAL_LEN);
		strcpy(att.stringVal, stringVal);
		delete stringVal;
	}
	return att;
}

Predicate Unmarshaller::decodeEventPredicate(char *source, int &index) {
	Predicate pred;
	pred.eventType = decodeInt(source, index);
	pred.constraintsNum = decodeInt(source, index);
	pred.constraints = new Constraint[pred.constraintsNum];
	for (int i=0; i<pred.constraintsNum; i++) {
		pred.constraints[i] = decodeConstraint(source, index);
	}
	pred.refersTo = decodeInt(source, index);
	pred.win = decodeLong(source, index);
	pred.kind = decodeCompKind(source, index);
	return pred;
}

ComplexParameter Unmarshaller::decodeParameter(char *source, int &index) {
	ComplexParameter par;
	par.operation = decodeConstraintOp(source, index);
	par.type = decodeStateType(source, index);
	par.vtype = decodeValType(source, index);
	par.leftTree = decodeOpTree(source, index);
	par.rightTree = decodeOpTree(source, index);
	return par;
}

Negation Unmarshaller::decodeNegation(char *source, int &index) {
	Negation neg;
	neg.eventType = decodeInt(source, index);
	neg.constraintsNum = decodeInt(source, index);
	neg.constraints = new Constraint[neg.constraintsNum];
	for (int i=0; i<neg.constraintsNum; i++) {
		neg.constraints[i] = decodeConstraint(source, index);
	}
	neg.lowerId = decodeInt(source, index);
	neg.lowerTime = decodeLong(source, index);
	neg.upperId = decodeInt(source, index);
	return neg;
}

Aggregate Unmarshaller::decodeAggregate(char *source, int &index) {
	Aggregate agg;
	agg.eventType = decodeInt(source, index);
	agg.constraintsNum = decodeInt(source, index);
	agg.constraints = new Constraint[agg.constraintsNum];
	for (int i=0; i<agg.constraintsNum; i++) {
		agg.constraints[i] = decodeConstraint(source, index);
	}
	agg.lowerId = decodeInt(source, index);
	agg.lowerTime = decodeLong(source, index);
	agg.upperId = decodeInt(source, index);
	agg.fun = decodeAggregateFun(source, index);
	char *name = decodeString(source, index);
	strcpy(agg.name, name);
	delete name;
	return agg;
}

CompositeEventTemplate * Unmarshaller::decodeEventTemplate(char *source, int &index) {
	int eventType = decodeInt(source, index);
	int attrNum = decodeInt(source, index);
	CompositeEventTemplate *eventTemplate = new CompositeEventTemplate(eventType);
	for (int i=0; i<attrNum; i++) {
		char *name = decodeString(source, index);
		OpTree *tree = decodeOpTree(source, index);
		eventTemplate->addAttribute(name, tree);
		delete name;
	}
	int staticAttrNum = decodeInt(source, index);
	for (int i=0; i<staticAttrNum; i++) {
		Attribute sAttr = decodeAttribute(source, index);
		eventTemplate->addStaticAttribute(sAttr);
	}

	return eventTemplate;
}

OpTree * Unmarshaller::decodeOpTree(char *source, int &index) {
	OpTreeType type = decodeOpTreeType(source, index);
	ValType valType = decodeValType(source, index);
	if (type == LEAF) {
		OpValueReference * val = decodeValueReference(source, index);
		return new OpTree(val, valType);
	} else {
		OpTree *leftTree = decodeOpTree(source, index);
		OpTree *rightTree = decodeOpTree(source, index);
		OpTreeOperation op = decodeOpTreeOperation(source, index);
		return new OpTree(leftTree, rightTree, op, valType);
	}
}

StaticValueReference * Unmarshaller::decodeStaticValueReference(char *source, int &index) {
	ValType type = decodeValType(source, index);
	if (type==INT) return new StaticValueReference(decodeInt(source, index));
	else if (type==FLOAT) return new StaticValueReference(decodeFloat(source, index));
	else if (type==BOOL) return new StaticValueReference(decodeBoolean(source, index));
	else /*if (type==STRING)*/ return new StaticValueReference(decodeString(source, index));
}

RulePktValueReference * Unmarshaller::decodeRulePktReference(char *source, int &index) {
	int idx = decodeInt(source, index);
	StateType type = decodeStateType(source, index);
	char *name = decodeString(source, index);
	//If name > 0 => it's an agg, but as a parameter, so that we need also an attribute name!
	if (type==AGG && strlen(name)==0) {
		delete name;
		return new RulePktValueReference(idx);
	}
	else {
		RulePktValueReference *ref = new RulePktValueReference(idx, name, type);
		delete name;
		return ref;
	}	
}

OpValueReference * Unmarshaller::decodeValueReference(char *source, int &index) {
	ValRefType vrtype = decodeValRefType(source, index);
	if (vrtype == RULEPKT) return decodeRulePktReference(source, index);
	else /*if (vrtype == STATIC)*/ return decodeStaticValueReference(source, index);
}

set<int> Unmarshaller::decodeIntSet(char *source, int &index) {
	int size = decodeInt(source, index);
	set<int> returnSet;
	for (int i=0; i<size; i++) {
		int element = decodeInt(source, index);
		returnSet.insert(element);
	}
	return returnSet;
}

bool Unmarshaller::decodeBoolean(char *source, int &index) {
	bool returnValue = (source[index++] == 1);
	return returnValue;
}

int Unmarshaller::decodeInt(char *source, int &index) {
	int returnValue = (0xff & source[index]) << 24 |
			(0xff & source[index+1]) << 16 |
			(0xff & source[index+2]) << 8  |
			(0xff & source[index+3]) << 0;
	index+=4;
	return returnValue;
}

float Unmarshaller::decodeFloat(char *source, int &index) {
	float returnValue = 0;
	int *ptr = (int *) &returnValue;
	*ptr = (0xff & source[index]) << 24 |
			(0xff & source[index+1]) << 16 |
			(0xff & source[index+2]) << 8  |
			(0xff & source[index+3]) << 0;
	index+=4;
	return returnValue;
}

long Unmarshaller::decodeLong(char *source, int &index) {
	long returnValue = (0xff & (long)source[index]) << 56 |
			(0xff & (long)source[index+1]) << 48 |
			(0xff & (long)source[index+2]) << 40 |
			(0xff & (long)source[index+3]) << 32 |
			(0xff & (long)source[index+4]) << 24 |
			(0xff & (long)source[index+5]) << 16 |
			(0xff & (long)source[index+6]) << 8	|
			(0xff & (long)source[index+7]) << 0;
	index+=8;
	return returnValue;
}

char * Unmarshaller::decodeString(char *source, int &index) {
	int length = decodeInt(source, index);
	char *charArray = new char[length+1];
	for (int i=0; i<length; i++) {
		charArray[i] = (char) source[index+i];
	}
	charArray[length] = '\0';
	index += length;
	return charArray;
}

CompKind Unmarshaller::decodeCompKind(char *source, int &index) {
	CompKind result;
	if (source[index]==0) result = EACH_WITHIN;
	else if (source[index]==1) result = FIRST_WITHIN;
	else if (source[index]==2) result = LAST_WITHIN;
	else result = ALL_WITHIN;
	index++;
	return result;
}

Op Unmarshaller::decodeConstraintOp(char *source, int &index) {
	Op result = EQ;
	if (source[index]==0) result = EQ;
	else if (source[index]==1) result = LT;
	else if (source[index]==2) result = GT;
	else if (source[index]==3) result = DF;
	else if (source[index]==4) result = IN;
	else if (source[index]==5) result = LE;
	else if (source[index]==6) result = GE;
	index++;
	return result;
}

StateType Unmarshaller::decodeStateType(char *source, int &index) {
	StateType result;
	if (source[index]==0) result = STATE;
	else if (source[index]==1) result = NEG;
	else result = AGG;
	index++;
	return result;
}

ValRefType Unmarshaller::decodeValRefType(char *source, int &index) {
	ValRefType result;
	if (source[index]==0) result = RULEPKT;
	else /*if (source[index]==1)*/ result = STATIC;
	index++;
	return result;
}

ValType Unmarshaller::decodeValType(char *source, int &index) {
	ValType result;
	if (source[index]==0) result = INT;
	else if (source[index]==1) result = FLOAT;
	else if (source[index]==2) result = BOOL;
	else result = STRING;
	index++;
	return result;
}

AggregateFun Unmarshaller::decodeAggregateFun(char *source, int &index) {
	AggregateFun result;
	if (source[index]==0) result = NONE;
	else if (source[index]==1) result = AVG;
	else if (source[index]==2) result = COUNT;
	else if (source[index]==3) result = MIN;
	else if (source[index]==4) result = MAX;
	else result = SUM;
	index++;
	return result;
}

OpTreeType Unmarshaller::decodeOpTreeType(char *source, int &index) {
	if (source[index++]==0) return LEAF;
	else return INNER;
}

OpTreeOperation Unmarshaller::decodeOpTreeOperation(char *source, int &index) {
	OpTreeOperation result;
	if (source[index]==0) result = ADD;
	else if (source[index]==1) result = SUB;
	else if (source[index]==2) result = MUL;
	else if (source[index]==3) result = DIV;
	else if (source[index]==4) result = AND;
	else result = OR;
	index++;
	return result;
}

PktType Unmarshaller::decodePktType(char *source, int &index) {
	PktType result;
	if (source[index]==0) result = PUB_PKT;
	else if (source[index]==1) result = RULE_PKT;
	else if (source[index]==2) result = SUB_PKT;
	else if (source[index]==3) result = ADV_PKT;
	else result = JOIN_PKT;
	index++;
	return result;
}
#pragma GCC pop_options
