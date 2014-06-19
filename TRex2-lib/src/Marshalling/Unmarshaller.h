//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara, Daniele Rogora
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

#ifndef UNMARSHALLER_H_
#define UNMARSHALLER_H_

#include <string.h>
#include "../Common/Consts.h"
#include "../Common/OpTree.h"
#include "../Packets/RulePkt.h"
#include "../Packets/PubPkt.h"
#include "../Packets/SubPkt.h"
#include "../Packets/AdvPkt.h"
#include "../Packets/JoinPkt.h"
#include "../Packets/RulePktValueReference.h"

/**
 * This file contains all the functions used to decode packets from arrays of bytes.
 */
class Unmarshaller {

public:

	RulePkt * decodeRulePkt(char *source);
	PubPkt * decodePubPkt(char *source);
	SubPkt * decodeSubPkt(char *source);
	AdvPkt * decodeAdvPkt(char *source);
	JoinPkt * decodeJoinPkt(char *source);
	int decodeInt(char *source);
	PktType decodePktType(char *source);

protected:

	RulePkt * decodeRulePkt(char *source, int &index);

	PubPkt * decodePubPkt(char *source, int &index);

	SubPkt * decodeSubPkt(char *source, int &index);

	AdvPkt * decodeAdvPkt(char *source, int &index);

	JoinPkt * decodeJoinPkt(char *source, int &index);

	Constraint decodeConstraint(char *source, int &index);

	Attribute decodeAttribute(char *source, int &index);

	Predicate decodeEventPredicate(char *source, int &index);

	ComplexParameter decodeParameter(char *source, int &index);

	Negation decodeNegation(char *source, int &index);

	Aggregate decodeAggregate(char *source, int &index);

	CompositeEventTemplate * decodeEventTemplate(char *source, int &index);

	OpTree * decodeOpTree(char *source, int &index);

	OpValueReference * decodeValueReference(char *source, int &index);
	
	RulePktValueReference * decodeRulePktReference(char *source, int &index);
	
	StaticValueReference * decodeStaticValueReference(char *source, int &index);

	std::set<int> decodeIntSet(char *source, int &index);

	bool decodeBoolean(char *source, int &index);

	int decodeInt(char *source, int &index);

	float decodeFloat(char *source, int &index);

	long decodeLong(char *source, int &index);

	char * decodeString(char *source, int &index);

	CompKind decodeCompKind(char *source, int &index);

	Op decodeConstraintOp(char *source, int &index);

	StateType decodeStateType(char *source, int &index);

	ValType decodeValType(char *source, int &index);
	
	ValRefType decodeValRefType(char *source, int &index);

	AggregateFun decodeAggregateFun(char *source, int &index);

	OpTreeType decodeOpTreeType(char *source, int &index);

	OpTreeOperation decodeOpTreeOperation(char *source, int &index);

	PktType decodePktType(char *source, int &index);

};

#endif /* UNMARSHALLER_H_ */
