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

#include "IndexingTable.h"

using namespace std;

IndexingTable::IndexingTable() { }

IndexingTable::~IndexingTable() {
	for (set<TablePred *>::iterator it=usedPreds.begin(); it!=usedPreds.end(); it++) {
		TablePred *tp = *it;
		delete tp;
	}
}

void IndexingTable::installRulePkt(RulePkt *pkt) {
	int numPreds = pkt->getPredicatesNum();
	// Iterating on rule predicates (states)
	for (int state=0; state<numPreds; state++) {
		Predicate pred = pkt->getPredicate(state);
		TablePred *tp = new TablePred;
		int eventType = pred.eventType;
		tp->ruleId = pkt->getRuleId();
		tp->stateId = state;
		tp->constraintsNum = pred.constraintsNum;
		tp->stateType = STATE;
		installTablePredicate(eventType, pred.constraints, pred.constraintsNum, tp);
	}
	// Iterating on rule aggregates
	int aggregatesNum = pkt->getAggregatesNum();
	for (int aggId=0; aggId<aggregatesNum; aggId++) {
		Aggregate agg = pkt->getAggregate(aggId);
		TablePred *tp = new TablePred;
		tp->ruleId = pkt->getRuleId();
		tp->stateId = aggId;
		tp->constraintsNum = agg.constraintsNum;
		tp->stateType = AGG;
		int eventType = agg.eventType;
		installTablePredicate(eventType, agg.constraints, agg.constraintsNum, tp);
	}
	// Iterating on rule negations
	int negationsNum = pkt->getNegationsNum();
	for (int negId=0; negId<negationsNum; negId++) {
		Negation neg = pkt->getNegation(negId);
		TablePred *tp = new TablePred;
		tp->ruleId = pkt->getRuleId();
		tp->stateId = negId;
		tp->constraintsNum = neg.constraintsNum;
		tp->stateType = NEG;
		int eventType = neg.eventType;
		installTablePredicate(eventType, neg.constraints, neg.constraintsNum, tp);
	}
}

void IndexingTable::installTablePredicate(int eventType, Constraint *constraints, int numConstraints, TablePred *tp) {
	usedPreds.insert(tp);
	if (numConstraints==0) {
		installNoConstraint(eventType, tp);
	}
	for (int i=0; i<numConstraints; i++) {
		installConstraint(eventType, constraints[i], tp);
	}
}

void IndexingTable::installNoConstraint(int eventType, TablePred *tp) {
	if (noIndex.find(eventType)==noIndex.end()) {
		NoConstraintIndex emptyIndex;
		noIndex.insert(make_pair(eventType, emptyIndex));
	}
	noIndex[eventType].installPredicate(tp);
}

void IndexingTable::installConstraint(int eventType, Constraint &c, TablePred *tp) {
	// Installing on the index of the appropriate type
	switch (c.type) {
	case INT:
		if (intIndex.find(eventType)==intIndex.end()) {
			IntConstraintIndex emptyIndex;
			intIndex.insert(make_pair(eventType, emptyIndex));
		}
		intIndex[eventType].installConstraint(c, tp);
		break;
	case FLOAT:
		if (floatIndex.find(eventType)==floatIndex.end()) {
			FloatConstraintIndex emptyIndex;
			floatIndex.insert(make_pair(eventType, emptyIndex));
		}
		floatIndex[eventType].installConstraint(c, tp);
		break;
	case BOOL:
		if (boolIndex.find(eventType)==boolIndex.end()) {
			BoolConstraintIndex emptyIndex;
			boolIndex.insert(make_pair(eventType, emptyIndex));
		}
		boolIndex[eventType].installConstraint(c, tp);
		break;
	case STRING:
		if (stringIndex.find(eventType)==stringIndex.end()) {
			StringConstraintIndex emptyIndex;
			stringIndex.insert(make_pair(eventType, emptyIndex));
		}
		stringIndex[eventType].installConstraint(c, tp);
		break;
	}
}

void IndexingTable::processMessage(PubPkt *pkt, MatchingHandler &mh) {
	int eventType = pkt->getEventType();
	// predCount stores intermediate results, it will be useful when more types will be available
	map<TablePred *, int> predCount;
	if (noIndex.find(eventType)!=noIndex.end()) noIndex[eventType].processMessage(pkt, mh, predCount);
	if (intIndex.find(eventType)!=intIndex.end()) intIndex[eventType].processMessage(pkt, mh, predCount);
	if (floatIndex.find(eventType)!=floatIndex.end()) floatIndex[eventType].processMessage(pkt, mh, predCount);
	if (boolIndex.find(eventType)!=boolIndex.end()) boolIndex[eventType].processMessage(pkt, mh, predCount);
	if (stringIndex.find(eventType)!=stringIndex.end()) stringIndex[eventType].processMessage(pkt, mh, predCount);
}
