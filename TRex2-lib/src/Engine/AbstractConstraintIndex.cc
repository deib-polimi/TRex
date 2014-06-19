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

#include "AbstractConstraintIndex.h"

using namespace std;

void AbstractConstraintIndex::addToMatchingHandler(MatchingHandler &mh, TablePred *tp) {
	// Predicate refers to states
	if (tp->stateType==STATE) {
		map<int, set<int> >::iterator it=mh.matchingStates.find(tp->ruleId);
		if (it==mh.matchingStates.end()) {
			set<int> states;
			states.insert(tp->stateId);
			mh.matchingStates.insert(make_pair(tp->ruleId, states));
		} else {
			it->second.insert(tp->stateId);
		}
	}
	// Predicate refers to aggregates
	else if (tp->stateType==AGG) {
		map<int, set<int> >::iterator it=mh.matchingAggregates.find(tp->ruleId);
		if (it==mh.matchingAggregates.end()) {
			set<int> aggs;
			aggs.insert(tp->stateId);
			mh.matchingAggregates.insert(make_pair(tp->ruleId, aggs));
		} else {
			it->second.insert(tp->stateId);
		}
	}
	// Predicate refers to negations
	else if (tp->stateType==NEG) {
		map<int, set<int> >::iterator it=mh.matchingNegations.find(tp->ruleId);
		if (it==mh.matchingNegations.end()) {
			set<int> negs;
			negs.insert(tp->stateId);
			mh.matchingNegations.insert(make_pair(tp->ruleId, negs));
		} else {
			it->second.insert(tp->stateId);
		}
	}
}
