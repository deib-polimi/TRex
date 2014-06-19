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

#include "BoolConstraintIndex.h"

using namespace std;

BoolConstraintIndex::BoolConstraintIndex() { }

BoolConstraintIndex::~BoolConstraintIndex() {
	for (set<BoolTableConstraint *>::iterator it=usedConstraints.begin(); it!=usedConstraints.end(); it++) {
		BoolTableConstraint *itc = *it;
		delete itc;
	}
}

void BoolConstraintIndex::installConstraint(Constraint &constraints, TablePred *predicate) {
	// Looks if the same constraint is already installed in the table
	BoolTableConstraint * itc = getConstraint(constraints);
	if (itc == NULL) {
		// If the constraint is not found, it creates a new one ...
		itc = createConstraint(constraints);
		// ... and installs it
		installConstraint(itc);
	}
	// In both cases connects the predicate with the constraint
	itc->connectedPredicates.insert(predicate);
}

void BoolConstraintIndex::processMessage(PubPkt *pkt, MatchingHandler &mh, map<TablePred *, int> &predCount) {
	for (int i=0; i<pkt->getAttributesNum(); i++) {
		if (pkt->getAttribute(i).type!=BOOL) continue;
		string name = pkt->getAttribute(i).name;
		bool val = pkt->getBoolAttributeVal(i);
		if (indexes.find(name)==indexes.end()) continue;
		// Equality constraints
		map<bool, BoolTableConstraint *>::iterator it = indexes[name].eq.find(val);
		if (it!=indexes[name].eq.end()) {
			BoolTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
		// Different from constraints
		it = indexes[name].df.find(!val);
		if (it!=indexes[name].df.end()) {
			BoolTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
	}
}

BoolTableConstraint * BoolConstraintIndex::getConstraint(Constraint &c) {
	for (set<BoolTableConstraint *>::iterator it=usedConstraints.begin(); it!=usedConstraints.end(); ++it) {
		BoolTableConstraint *itc = *it;
		if (itc->op!=c.op) continue;
		if (itc->val!=c.boolVal) continue;
		if (strcmp(itc->name, c.name)!=0) continue;
		return (itc);
	}
	return NULL;
}

BoolTableConstraint * BoolConstraintIndex::createConstraint(Constraint &c) {
	BoolTableConstraint *itc = new BoolTableConstraint;
	strcpy(itc->name, c.name);
	itc->op = c.op;
	itc->val = c.boolVal;
	return itc;
}

inline void BoolConstraintIndex::installConstraint(BoolTableConstraint *c) {
	usedConstraints.insert(c);
	string s = c->name;
	if (indexes.find(s)==indexes.end()) {
		BoolOperatorsTable emptyTable;
		indexes.insert(make_pair(s, emptyTable));
	}
	if (c->op==EQ) indexes[s].eq.insert(make_pair(c->val, c));
	else indexes[s].df.insert(make_pair(c->val, c));
}

inline void BoolConstraintIndex::processConstraint(BoolTableConstraint *c, MatchingHandler &mh, map<TablePred *, int> &predCount) {
	for (set<TablePred *>::iterator it=c->connectedPredicates.begin(); it!=c->connectedPredicates.end(); ++it) {
		// If satisfied for the first time, sets count to 1
		if (predCount.find(*it)==predCount.end()) predCount.insert(make_pair(*it, 1));
		// Otherwise increases count by one
		else ++predCount[*it];
		if (predCount[*it] == (*it)->constraintsNum) addToMatchingHandler(mh, (*it));
	}
}
