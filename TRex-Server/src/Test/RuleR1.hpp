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

#ifndef RULER1_H_
#define RULER1_H_

#include "../external.hpp"
#include "TestRule.hpp"

namespace concept{
namespace test{

/*
 * Rule R1:
 *
 * define	Fire(area: string, measuredTemp: int)
 * from		Smoke(area=$a) and
 * 			each Temp(area=$a and value>45) within 5 min. from Smoke
 * where	area=Smoke.area and measuredTemp=Temp.value
 */
class RuleR1: public TestRule{
public:
	RulePkt* buildRule();
	SubPkt* buildSubscription();
	vector<PubPkt*> buildPublication();
};

} // test
} // concept

#endif /* RULER1_H_ */
