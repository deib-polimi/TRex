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

#include "TimeMs.h"

using namespace std;

TimeMs::TimeMs() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  timeVal = (uint64_t)tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

TimeMs::TimeMs(const TimeMs& x) { timeVal = x.timeVal; }

TimeMs::TimeMs(uint64_t partimeVal) { timeVal = partimeVal; }

TimeMs::~TimeMs() {
  // Nothing to do
}

bool TimeMs::elapsed() const {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  uint64_t currentTimeVal = (uint64_t)tp.tv_sec * 1000 + tp.tv_usec / 1000;
  return timeVal < currentTimeVal;
}

uint64_t TimeMs::getTimeVal() const { return timeVal; }

TimeMs TimeMs::operator+(const TimeMs& x) {
  return TimeMs(x.timeVal + timeVal);
}

TimeMs TimeMs::operator-(const TimeMs& x) {
  if (timeVal > x.timeVal)
    return TimeMs(timeVal - x.timeVal);
  return TimeMs(0);
}

TimeMs& TimeMs::operator=(const TimeMs& x) {
  timeVal = x.timeVal;
  return *this;
}

TimeMs& TimeMs::operator+=(const TimeMs& x) {
  timeVal += x.timeVal;
  return *this;
}

TimeMs& TimeMs::operator-=(const TimeMs& x) {
  timeVal -= x.timeVal;
  return *this;
}

bool TimeMs::operator==(const TimeMs& x) const { return timeVal == x.timeVal; }

bool TimeMs::operator!=(const TimeMs& x) const { return timeVal != x.timeVal; }

bool TimeMs::operator<(const TimeMs& x) const { return timeVal < x.timeVal; }

bool TimeMs::operator>(const TimeMs& x) const { return timeVal > x.timeVal; }

bool TimeMs::operator<=(const TimeMs& x) const { return timeVal <= x.timeVal; }

bool TimeMs::operator>=(const TimeMs& x) const { return timeVal >= x.timeVal; }

ostream& operator<<(ostream& out, const TimeMs& x) {
  out << x.getTimeVal();
  return out;
}
