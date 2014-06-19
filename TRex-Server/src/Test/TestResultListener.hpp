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

#ifndef TESTRESULTLISTENER_H_
#define TESTRESULTLISTENER_H_

#include "../external.hpp"
#include "../util.hpp"

namespace concept{
namespace test{

class TestResultListener: public ResultListener{
public:
	TestResultListener(SubPkt* subscription);
	virtual ~TestResultListener();

	virtual void handleResult(set<PubPkt *> &genPkts, double procTime);
	int getId() {return id;}
private:
	SubPkt* subscription;
	int id;
	void printMessage(std::string msg);
};

} // test
} // concept

#endif /* TESTRESULTLISTENER_H_ */
