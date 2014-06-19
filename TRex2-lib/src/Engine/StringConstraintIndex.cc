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

#include "StringConstraintIndex.h"

using namespace std;

StringConstraintIndex::StringConstraintIndex() { }

StringConstraintIndex::~StringConstraintIndex() {
	for (set<StringTableConstraint *>::iterator it=usedConstraints.begin(); it!=usedConstraints.end(); it++) {
		StringTableConstraint *itc = *it;
		delete itc;
	}
}

void StringConstraintIndex::installConstraint(Constraint &constraints, TablePred *predicate) {
	// Looks if the same constraint is already installed in the table
	StringTableConstraint * itc = getConstraint(constraints);
	if (itc == NULL) {
		// If the constraint is not found, it creates a new one ...
		itc = createConstraint(constraints);
		// ... and installs it
		installConstraint(itc);
	}
	// In both cases connects the predicate with the constraint
	itc->connectedPredicates.insert(predicate);
}

void StringConstraintIndex::processMessage(PubPkt *pkt, MatchingHandler &mh, map<TablePred *, int> &predCount) {
	for (int i=0; i<pkt->getAttributesNum(); i++) {
		if (pkt->getAttribute(i).type!=STRING) continue;
		string name = pkt->getAttribute(i).name;
		string val = pkt->getAttribute(i).stringVal;
		if (indexes.find(name)==indexes.end()) continue;
		// Equality constraints
		map<string, StringTableConstraint *>::iterator it = indexes[name].eq.find(val);
		if (it!=indexes[name].eq.end()) {
			StringTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
		// Include constraints
		for (it=indexes[name].in.begin(); it!=indexes[name].in.end(); ++it) {
			string storedVal = it->first;
			if (val.find(storedVal)==string::npos) continue;
			StringTableConstraint *itc = it->second;
			processConstraint(itc, mh, predCount);
		}
	}
}

StringTableConstraint * StringConstraintIndex::getConstraint(Constraint &c) {
	for (set<StringTableConstraint *>::iterator it=usedConstraints.begin(); it!=usedConstraints.end(); ++it) {
		StringTableConstraint *itc = *it;
		if (itc->op!=c.op) continue;
		if (itc->val==c.stringVal) continue;
		if (strcmp(itc->name, c.name)!=0) continue;
		return (itc);
	}
	return NULL;
}

StringTableConstraint * StringConstraintIndex::createConstraint(Constraint &c) {
	StringTableConstraint *itc = new StringTableConstraint;
	strcpy(itc->name, c.name);
	itc->op = c.op;
	itc->val = c.stringVal;
	return itc;
}

inline void StringConstraintIndex::installConstraint(StringTableConstraint *c) {
	usedConstraints.insert(c);
	string s = c->name;
	string val = c->val;
	if (indexes.find(s)==indexes.end()) {
		StringOperatorsTable emptyTable;
		indexes.insert(make_pair(s, emptyTable));
	}
	if (c->op==EQ) indexes[s].eq.insert(make_pair(val, c));
	else if (c->op==IN) indexes[s].in.insert(make_pair(val, c));
}

inline void StringConstraintIndex::processConstraint(StringTableConstraint *c, MatchingHandler &mh, map<TablePred *, int> &predCount) {
	for (set<TablePred *>::iterator it=c->connectedPredicates.begin(); it!=c->connectedPredicates.end(); ++it) {
		// If satisfied for the first time, sets count to 1
		if (predCount.find(*it)==predCount.end()) predCount.insert(make_pair(*it, 1));
		// Otherwise increases count by one
		else ++predCount[*it];
		if (predCount[*it] == (*it)->constraintsNum) addToMatchingHandler(mh, (*it));
	}
}
