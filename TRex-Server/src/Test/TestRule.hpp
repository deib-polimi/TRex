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

#ifndef TESTRULE_H_
#define TESTRULE_H_

#include "../external.hpp"

namespace concept{
namespace test{

/**
 * A test case for a particular TESLA rule, to be tested with T-Rex.
 */
class TestRule{
public:
	// Event types
	static const int EVENT_POSITION = 1;
	static const int EVENT_STOPPEDPOSITION = 2;
	static const int EVENT_GATHERING = 3;
	static const int EVENT_SMOKE = 10;
	static const int EVENT_TEMP = 11;
	static const int EVENT_FIRE= 12;

	virtual ~TestRule() {};

	/**
	 * Generates the rule
	 */
	virtual RulePkt* buildRule() =0;

	/**
	 * Generates a possible subscription to this rule
	 */
	virtual SubPkt* buildSubscription() =0;

	/**
	 * Generates a possible sequence of "simple" events to be published in T-Rex,
	 * that should cause the creation of the "complex" event referred by this rule,
	 * according to the subscription made with buildSubscription().
	 * The PubPkt(s) should be published in T-Rex in the order in which they are contained
	 * in the vector.
	 */
	virtual vector<PubPkt*> buildPublication() =0;

protected:
	Constraint NO_CONSTRAINTS[0];
};

} // test
} // concept

#endif /* TESTRULE_H_ */
