/*
 * Copyright (C) 2011 Francesco Feltrinelli <first_name DOT last_name AT gmail DOT com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "RuleR1.hpp"
#include "RuleR0.hpp"

using namespace concept::test;

RulePkt* RuleR1::buildRule(){
	RulePkt* rule= new RulePkt(false);

	int indexPredSmoke= 0;
	int indexPredTemp= 1;

	TimeMs fiveMin(1000*60*5);

	// Smoke root predicate
	// Fake constraint as a temporary workaround to an engine's bug
	// FIXME remove workaround when bug fixed
	Constraint fakeConstr[1];
	strcpy(fakeConstr[0].name, RuleR0::ATTR_AREA);
	fakeConstr[0].type= STRING;
	fakeConstr[0].op = IN;
	strcpy(fakeConstr[0].stringVal, "");
	rule->addRootPredicate(EVENT_SMOKE, fakeConstr, 1);

	// Temp predicate
	// Constraint: Temp.value > 45
	Constraint tempConstr[1];
	strcpy(tempConstr[0].name, RuleR0::ATTR_TEMPVALUE);
	tempConstr[0].type= INT;
	tempConstr[0].op= GT;
	tempConstr[0].intVal= 45;
	rule->addPredicate(EVENT_TEMP, tempConstr, 1, indexPredSmoke, fiveMin, EACH_WITHIN);

	// Parameter: Smoke.area=Temp.area
	rule->addParameterBetweenStates(indexPredSmoke, RuleR0::ATTR_AREA, indexPredTemp, RuleR0::ATTR_AREA);

	// Fire template
	CompositeEventTemplate* fireTemplate= new CompositeEventTemplate(EVENT_FIRE);
	// Area attribute in template
	OpTree* areaOpTree= new OpTree(new RulePktValueReference(indexPredSmoke, RuleR0::ATTR_AREA, STATE), STRING);
	fireTemplate->addAttribute(RuleR0::ATTR_AREA, areaOpTree);
	// MeasuredTemp attribute in template
	OpTree* measuredTempOpTree= new OpTree(new RulePktValueReference(indexPredTemp, RuleR0::ATTR_TEMPVALUE, STATE), INT);
	fireTemplate->addAttribute(RuleR0::ATTR_MEASUREDTEMP, measuredTempOpTree);

	rule->setCompositeEventTemplate(fireTemplate);

	return rule;
}

SubPkt* RuleR1::buildSubscription() {
	Constraint constr[1];
	// Area constraint
	strcpy(constr[0].name, RuleR0::ATTR_AREA);
	constr[0].type= STRING;
	constr[0].op= EQ;
	strcpy(constr[0].stringVal, RuleR0::AREA_OFFICE);

	return new SubPkt(EVENT_FIRE, constr, 1);
}

vector<PubPkt*> RuleR1::buildPublication(){

	// Temp event
	Attribute tempAttr[2];
	// Value attribute
	strcpy(tempAttr[0].name, RuleR0::ATTR_TEMPVALUE);
	tempAttr[0].type= INT;
	tempAttr[0].intVal= 50;
	// Area attribute
	strcpy(tempAttr[1].name, RuleR0::ATTR_AREA);
	tempAttr[1].type= STRING;
	strcpy(tempAttr[1].stringVal, RuleR0::AREA_OFFICE);
	PubPkt* tempPubPkt= new PubPkt(EVENT_TEMP, tempAttr, 2);

	// Smoke event
	// Area attribute
	Attribute smokeAttr[1];
	strcpy(smokeAttr[0].name, RuleR0::ATTR_AREA);
	smokeAttr[0].type= STRING;
	strcpy(smokeAttr[0].stringVal, RuleR0::AREA_OFFICE);
	PubPkt* smokePubPkt= new PubPkt(EVENT_SMOKE, smokeAttr, 1);

	vector<PubPkt*> pubPkts;
	pubPkts.push_back(tempPubPkt);
	pubPkts.push_back(smokePubPkt);
	return pubPkts;
}
