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

#ifndef TIMEMS_H_
#define TIMEMS_H_

#include <sys/time.h>
#include <stdint.h>
#include <iostream>

/**
 * This class represents time in millisecond and implements all methods to
 * manage time: add, subtract, compare and assign.
 * Even if time is represented in millisecond, there are no guarantees about
 * actual precision. This is system dependent.
 */
class TimeMs {
public:

	/**
	 * Empty constructor: builds a new TimeMs with the current time value
	 */
	TimeMs();

	/**
	 * Copy constructor
	 */
	TimeMs(const TimeMs &x);

	/**
	 * Constructor with time value
	 */
	TimeMs(uint64_t parTime);

	/**
	 * Destructor
	 */
	virtual ~TimeMs();

	/**
	 * Return true if and only if the time has elapsed
	 */
	bool elapsed() const;

	/**
	 * Returns the stored value in milliseconds
	 */
	uint64_t getTimeVal() const;

	/**
	 * Overriding of operators
	 */
	TimeMs operator+(const TimeMs &x);
	TimeMs operator-(const TimeMs &x);
	TimeMs & operator=(const TimeMs &x);
	TimeMs & operator+=(const TimeMs &x);
	TimeMs & operator-=(const TimeMs &x);
	bool operator==(const TimeMs &x) const;
	bool operator!=(const TimeMs &x) const;
	bool operator<(const TimeMs &x) const;
	bool operator>(const TimeMs &x) const;
	bool operator<=(const TimeMs &x) const;
	bool operator>=(const TimeMs &x) const;

private:
	// Time value
	 uint64_t timeVal;

};

std::ostream& operator<<(std::ostream &out, const TimeMs &x);

#endif
