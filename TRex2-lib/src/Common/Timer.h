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

#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

/**
 * This is a timer with the precision of microseconds: it works on UNIX systems.
 */
class Timer {

public:

		/**
     * Empty constructoy
     */
		Timer();

		/**
		 * Destructor
		 */
		~Timer();

    /**
     * Start the timer
     */
		void   start();

		/**
		 * Stop the timer
		 */
		void   stop();

    /**
     * Get elapsed time in seconds
     */
    double getElapsedTimeInSec();

    /**
     * Get elapsed time in milliseconds
     */
    double getElapsedTimeInMilliSec();

    /**
     * Get the elapsed time in microseconds
     */
    double getElapsedTimeInMicroSec();

private:
    double startTimeInMicroSec;					// starting time in micro-second
    double endTimeInMicroSec;						// ending time in micro-second
    int stopped;												// stop flag
    timeval startCount;
    timeval endCount;
};

#endif /* TIMER_H */
