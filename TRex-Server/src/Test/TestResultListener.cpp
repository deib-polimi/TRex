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

#include "TestResultListener.hpp"

using namespace concept::test;
using namespace concept::util;

static int lastId= 0;

TestResultListener::TestResultListener(SubPkt* subscription):
	subscription(subscription), id(lastId++)
{}

TestResultListener::~TestResultListener(){
	delete subscription;
}

void TestResultListener::handleResult(set<PubPkt *> &genPkts, double procTime){
	for (set<PubPkt*>::iterator i= genPkts.begin(); i != genPkts.end(); i++){
		PubPkt* pubPkt= *i;
		printMessage("New complex event created:");
		printMessage(toString(pubPkt));
		if (matches(subscription, pubPkt)){
			printMessage("My subscription is matched");
		} else {
			printMessage("My subscription is not matched");
		}
	}
}

void TestResultListener::printMessage(std::string msg){
	cout << "TestResultListener" << id << " > " << msg << endl;
}
