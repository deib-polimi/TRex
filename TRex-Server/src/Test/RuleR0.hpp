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

#ifndef RULER0_H_
#define RULER0_H_

#include "../external.hpp"
#include "TestRule.hpp"

namespace concept{
namespace test{

/*
 * Rule R0:
 *
 * define	Fire(area: string, measuredTemp: int)
 * from		Temp(value>45)
 * where	area=Temp.area and measuredTemp=Temp.value
 *
 */
class RuleR0: public TestRule{
public:
	RulePkt* buildRule();
	SubPkt* buildSubscription();
	vector<PubPkt*> buildPublication();

	// Attribute names
	static char ATTR_TEMPVALUE[];
	static char ATTR_AREA[];
	static char ATTR_MEASUREDTEMP[];

	// Possible values for attribute "area"
	static char AREA_GARDEN[];
	static char AREA_OFFICE[];
	static char AREA_TOILET[];
};

} // test
} // concept

#endif /* RULER0_H_ */
