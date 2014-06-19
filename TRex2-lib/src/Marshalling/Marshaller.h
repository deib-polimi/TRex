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

#ifndef MARSHALLER_H_
#define MARSHALLER_H_

#include <string.h>
#include "../Common/Consts.h"
#include "../Common/Structures.h"
#include "../Common/OpTree.h"
#include "../Packets/RulePkt.h"
#include "../Packets/PubPkt.h"
#include "../Packets/SubPkt.h"
#include "../Packets/AdvPkt.h"
#include "../Packets/JoinPkt.h"
#include "../Packets/RulePktValueReference.h"

/**
 * This file contains all the functions used to encode packets as array of bytes.
 */
class Marshaller {

public:

	char * encode(PubPkt *pkt);
	char * encode(RulePkt *pkt);
	char * encode(SubPkt *pkt);
	char * encode(AdvPkt *pkt);
	char * encode(JoinPkt *pkt);
	int getSize(PubPkt *source);
	int getSize(RulePkt *source);
	int getSize(SubPkt *source);
	int getSize(AdvPkt *source);
	int getSize(JoinPkt *source);

protected:

	int getNumBytes(RulePkt *source);
	int encode(RulePkt *source, char *dest, int startIndex);

	int getNumBytes(PubPkt *source);
	int encode(PubPkt *source, char *dest, int startIndex);

	int getNumBytes(SubPkt *source);
	int encode(SubPkt *source, char *dest, int startIndex);

	int getNumBytes(AdvPkt *source);
	int encode(AdvPkt *source, char *dest, int startIndex);

	int getNumBytes(JoinPkt *source);
	int encode(JoinPkt *source, char *dest, int startIndex);

	int getNumBytes(Constraint &source);
	int encode(Constraint &source, char *dest, int startIndex);

	int getNumBytes(Attribute &source);
	int encode(Attribute &source, char *dest, int startIndex);

	int getNumBytes(Predicate &source);
	int encode(Predicate &source, char *dest, int startIndex);

	int getNumBytes(Parameter &source);
	int encode(Parameter &source, char *dest, int startIndex);

	int getNumBytes(Negation &source);
	int encode(Negation &source, char *dest, int startIndex);

	int getNumBytes(Aggregate &source);
	int encode(Aggregate &source, char *dest, int startIndex);

	int getNumBytes(CompositeEventTemplate &source);
	int encode(CompositeEventTemplate &source, char *dest, int startIndex);

	int getNumBytes(CompositeEventTemplateAttr &source);
	int encode(CompositeEventTemplateAttr &source, char *dest, int startIndex);

	int getNumBytes(OpTree *source);
	int encode(OpTree *source, char *dest, int startIndex);

	int getNumBytes(RulePktValueReference *source);
	int encode(RulePktValueReference *source, char *dest, int startIndex);

	int getNumBytes(std::set<int> &source);
	int encode(std::set<int> &source, char *dest, int startIndex);

	int getNumBytes(bool source);
	int encode(bool source, char *dest, int startIndex);

	int getNumBytes(int source);
	int encode(int source, char *dest, int startIndex);
	
	int getNumBytes(uint64_t source);
	int encode(uint64_t source, char *dest, int startIndex);

	int getNumBytes(float source);
	int encode(float source, char *dest, int startIndex);

	int getNumBytes(long source);
	int encode(long source, char *dest, int startIndex);

	int getNumBytes(char *source);
	int encode(char *source, char *dest, int startIndex);

	int getNumBytes(CompKind source);
	int encode(CompKind source, char *dest, int startIndex);

	int getNumBytes(Op source);
	int encode(Op source, char *dest, int startIndex);

	int getNumBytes(ValType source);
	int encode(ValType source, char *dest, int startIndex);

	int getNumBytes(StateType source);
	int encode(StateType source, char *dest, int startIndex);

	int getNumBytes(AggregateFun source);
	int encode(AggregateFun source, char *dest, int startIndex);

	int getNumBytes(OpTreeType source);
	int encode(OpTreeType source, char *dest, int startIndex);

	int getNumBytes(OpTreeOperation source);
	int encode(OpTreeOperation source, char *dest, int startIndex);

	int getNumBytes(PktType source);
	int encode(PktType source, char *dest, int startIndex);

	int getNumBytes(TimeMs source);
	int encode(TimeMs source, char *dest, int startIndex);

};

#endif /* MARSHALLER_H_ */
