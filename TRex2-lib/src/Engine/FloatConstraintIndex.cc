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

#include "FloatConstraintIndex.h"

using namespace std;

FloatConstraintIndex::FloatConstraintIndex() { }

FloatConstraintIndex::~FloatConstraintIndex() {
	for (set<FloatTableConstraint *>::iterator it=usedConstraints.begin(); it!=usedConstraints.end(); it++) {
		FloatTableConstraint *itc = *it;
		delete itc;
	}
}

void FloatConstraintIndex::installConstraint(Constraint &constraints, TablePred *predicate) {
	// Looks if the same constraint is already installed in the table
	FloatTableConstraint * itc = getConstraint(constraints);
	if (itc == NULL) {
		// If the constraint is not found, it creates a new one ...
		itc = createConstraint(constraints);
		// ... and installs it
		installConstraint(itc);
	}
	// In both cases connects the predicate with the constraint
	itc->connectedPredicates.insert(predicate);
}

void FloatConstraintIndex::processMessage(PubPkt *pkt, MatchingHandler &mh, map<TablePred *, int> &predCount) {
	for (int i=0; i<pkt->getAttributesNum(); i++) {
		if (pkt->getAttribute(i).type!=FLOAT) continue;
		string name = pkt->getAttribute(i).name;
		float val = pkt->getFloatAttributeVal(i);
		if (indexes.find(name)==indexes.end()) continue;
		// Equality constraints
		map<float, FloatTableConstraint *>::iterator it = indexes[name].eq.find(val);
		if (it!=indexes[name].eq.end()) {
			FloatTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
		// Less then constraints (iterating in descending order)
		for (map<float, FloatTableConstraint *>::reverse_iterator rit=indexes[name].lt.rbegin(); rit!=indexes[name].lt.rend(); ++rit) {
			if (rit->first <= val) break;
			FloatTableConstraint *itc = rit->second;
			processConstraint(itc, mh, predCount);
		}
		// Greater than constraints (iterating in ascending order)
		for (it=indexes[name].gt.begin(); it!=indexes[name].gt.end(); ++it) {
			if (it->first >= val) break;
			FloatTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
		// Different from constraints (iterating in ascending order)
		for (it=indexes[name].df.begin(); it!=indexes[name].df.end(); ++it) {
			if (it->first == val) continue;
			FloatTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
	}
}

FloatTableConstraint * FloatConstraintIndex::getConstraint(Constraint &c) {
	for (set<FloatTableConstraint *>::iterator it=usedConstraints.begin(); it!=usedConstraints.end(); ++it) {
		FloatTableConstraint *itc = *it;
		if (itc->op!=c.op) continue;
		if (itc->val!=c.floatVal) continue;
		if (strcmp(itc->name, c.name)!=0) continue;
		return (itc);
	}
	return NULL;
}

FloatTableConstraint * FloatConstraintIndex::createConstraint(Constraint &c) {
	FloatTableConstraint *itc = new FloatTableConstraint;
	strcpy(itc->name, c.name);
	itc->op = c.op;
	itc->val = c.floatVal;
	return itc;
}

inline void FloatConstraintIndex::installConstraint(FloatTableConstraint *c) {
	usedConstraints.insert(c);
	string s = c->name;
	if (indexes.find(s)==indexes.end()) {
		FloatOperatorsTable emptyTable;
		indexes.insert(make_pair(s, emptyTable));
	}
	if (c->op==EQ) indexes[s].eq.insert(make_pair(c->val, c));
	else if (c->op==LT) indexes[s].lt.insert(make_pair(c->val, c));
	else if (c->op==GT) indexes[s].gt.insert(make_pair(c->val, c));
	else indexes[s].df.insert(make_pair(c->val, c));
}

inline void FloatConstraintIndex::processConstraint(FloatTableConstraint *c, MatchingHandler &mh, map<TablePred *, int> &predCount) {
	for (set<TablePred *>::iterator it=c->connectedPredicates.begin(); it!=c->connectedPredicates.end(); ++it) {
		// If satisfied for the first time, sets count to 1
		if (predCount.find(*it)==predCount.end()) predCount.insert(make_pair(*it, 1));
		// Otherwise increases count by one
		else ++predCount[*it];
		if (predCount[*it] == (*it)->constraintsNum) addToMatchingHandler(mh, (*it));
	}
}
