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

#include "Timer.h"
#include <stdlib.h>

Timer::Timer() {
	startCount.tv_sec = startCount.tv_usec = 0;
	endCount.tv_sec = endCount.tv_usec = 0;
	stopped = 0;
	startTimeInMicroSec = 0;
	endTimeInMicroSec = 0;
}

Timer::~Timer() {
	// Nothing to do
}

void Timer::start() {
	stopped = 0; // reset stop flag
	gettimeofday(&startCount, NULL);
}

void Timer::stop() {
	stopped = 1; // set timer stopped flag
	gettimeofday(&endCount, NULL);
}

double Timer::getElapsedTimeInMicroSec() {
	if(!stopped) gettimeofday(&endCount, NULL);
	startTimeInMicroSec = (startCount.tv_sec * 1000000.0) + startCount.tv_usec;
	endTimeInMicroSec = (endCount.tv_sec * 1000000.0) + endCount.tv_usec;
	return endTimeInMicroSec - startTimeInMicroSec;
}

double Timer::getElapsedTimeInMilliSec() {
	return this->getElapsedTimeInMicroSec() * 0.001;
}

double Timer::getElapsedTimeInSec() {
	return this->getElapsedTimeInMicroSec() * 0.000001;
}
