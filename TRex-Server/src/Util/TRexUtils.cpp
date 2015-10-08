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

#include "TRexUtils.hpp"

using namespace concept;
using namespace std;

bool util::matches(SubPkt* sub, PubPkt* pub){
	if (pub->getEventType()!=sub->getEventType()) return false;

	for (int i=0; i<sub->getConstraintsNum(); i++){
		Constraint constr= sub->getConstraint(i);
		int attrIndex;
		ValType valType;
		if (!pub->getAttributeIndexAndType(constr.name, attrIndex, valType)) return false;
		if (!matches(constr, pub->getAttribute(attrIndex))) return false;
	}
	return true;
}

bool util::matches(const Constraint& constr, const Attribute& attr){
	if (strcmp(constr.name, attr.name)!=0) return false;
	if (constr.type != attr.type) return false;

	switch (constr.type){
	case INT:
		switch (constr.op) {
		case EQ:
			if (attr.intVal!=constr.intVal) return false;
			break;
		case LT:
			if (attr.intVal>=constr.intVal) return false;
			break;
		case GT:
			if (attr.intVal<=constr.intVal) return false;
			break;
		case LE:
			if (attr.intVal>constr.intVal) return false;
			break;
		case GE:
			if (attr.intVal<constr.intVal) return false;
			break;
		
		case NE:
			if (attr.intVal==constr.intVal) return false;
			break;
		case IN:
			throw invalid_argument("IN operation cannot be used with INT type");
			break;
		}
		break;
	case FLOAT:
		switch (constr.op) {
		case EQ:
			if (attr.floatVal!=constr.floatVal) return false;
			break;
		case LT:
			if (attr.floatVal>=constr.floatVal) return false;
			break;
		case GT:
			if (attr.floatVal<=constr.floatVal) return false;
			break;
		case LE:
			if (attr.floatVal>constr.floatVal) return false;
			break;
		case GE:
			if (attr.floatVal<constr.floatVal) return false;
			break;

		case NE:
			if (attr.floatVal==constr.floatVal) return false;
			break;
		case IN:
			throw invalid_argument("IN operation cannot be used with FLOAT type");
			break;
		}
		break;
	case BOOL:
		switch (constr.op) {
		case EQ:
			if (attr.boolVal!=constr.boolVal) return false;
			break;
		case LT:
			throw invalid_argument("LT operation cannot be used with BOOL type");
			break;
		case GT:
			throw invalid_argument("GT operation cannot be used with BOOL type");
			break;
		case LE:
			throw invalid_argument("LE operation cannot be used with BOOL type");
			break;
		case GE:
			throw invalid_argument("GE operation cannot be used with BOOL type");
			break;

		case NE:
			if (attr.boolVal==constr.boolVal) return false;
			break;
		case IN:
			throw invalid_argument("IN operation cannot be used with BOOL type");
			break;
		}
		break;
	case STRING:
		switch (constr.op) {
		case EQ:
			if (strcmp(attr.stringVal, constr.stringVal)!=0) return false;
			break;
		case LT:
			if (strcmp(attr.stringVal, constr.stringVal)>=0) return false;
			break;
		case GT:
			if (strcmp(attr.stringVal, constr.stringVal)<=0) return false;
			break;
		case NE:
			if (strcmp(attr.stringVal, constr.stringVal)==0) return false;
			break;
		case LE:
			if (strcmp(attr.stringVal, constr.stringVal)>0) return false;
			break;
		case GE:
			if (strcmp(attr.stringVal, constr.stringVal)<0) return false;
			break;
			
		case IN:
			// The constraint's value should be a substring of the attribute's value:
			// it is a filter specified for published events' attributes
			if (strstr(attr.stringVal, constr.stringVal)==NULL) return false;
			break;
		}
		break;
	}

	return true;
}

bool util::equals(SubPkt* pkt1, SubPkt* pkt2){
	if (pkt1->getEventType() != pkt2->getEventType()) return false;
	if (pkt1->getConstraintsNum() != pkt2->getConstraintsNum()) return false;
	int constrNum= pkt1->getConstraintsNum();
	for (int i=0; i<constrNum; i++){
		bool found= false;
		for (int j=0; j<constrNum; j++){
			if (pkt1->getConstraint(i) == pkt2->getConstraint(j)){
				found= true;
				break;
			}
		}
		if (!found) return false;
	}
	return true;
}

std::string util::toString(PubPkt* pkt){
	stringstream ss;
	ss << "{event=" << pkt->getEventType() << ", attributes={";
	int attrNum= pkt->getAttributesNum();
	for (int i=0; i<= attrNum-1; i++){
		ss << toString(pkt->getAttribute(i));
		if (i<attrNum-1) ss << ", ";
		else ss << "}";
	}
	ss << "}";
	return ss.str();
}

std::string util::toString(RulePkt* pkt){
	CompositeEventTemplate *templ= pkt->getCompositeEventTemplate();
	stringstream ss;
	ss << "{event=" << templ->getEventType() << ", attributes={";
	int attrNum= templ->getAttributesNum();
	for (int i=0; i<= attrNum-1; i++){
		ss << templ->getAttributeName(i);
		if (i<attrNum-1) ss << ", ";
		else ss << "}";
	}
	ss << "}";
	return ss.str();
}

std::string util::toString(SubPkt* pkt){
	stringstream ss;
	ss << "{event=" << pkt->getEventType() << ", constraints={";
	int constrNum= pkt->getConstraintsNum();
	for (int i=0; i<= constrNum-1; i++){
		ss << toString(pkt->getConstraint(i));
		if (i<constrNum-1) ss << ", ";
		else ss << "}";
	}
	ss << "}";
	return ss.str();
}

std::string util::toString(const Attribute& attr){
	stringstream ss;
	ss << attr.name << "=";

	switch (attr.type){
	case INT:
		ss << attr.intVal;
		break;
	case FLOAT:
		ss << attr.floatVal;
		break;
	case BOOL:
		ss << attr.boolVal;
		break;
	case STRING:
		ss << '"' << attr.stringVal << '"';
		break;
	}

	return ss.str();
}

std::string util::toString(const Constraint& constr){
	stringstream ss;
	ss << constr.name;

	switch (constr.op) {
	case EQ:
		ss << "=";
		break;
	case LT:
		ss << "<";
		break;
	case GT:
		ss << ">";
		break;
	case NE:
		ss << "!=";
		break;
	case LE:
		ss << "<=";
		break;
	case GE:
		ss << ">=";
		break;

	case IN:
		// set-membership ASCII symbol for Event-B language is used here for substrings
		// see http://wiki.event-b.org/index.php/ASCII_Representations_of_the_Mathematical_Symbols_%28Rodin_User_Manual%29
		ss << ":";
		break;
	}

	switch (constr.type){
	case INT:
		ss << constr.intVal;
		break;
	case FLOAT:
		ss << constr.floatVal;
		break;
	case BOOL:
		ss << constr.boolVal;
		break;
	case STRING:
		ss << '"' << constr.stringVal << '"';
		break;
	}
	return ss.str();
}

std::string util::toString(const boost::asio::ip::tcp::socket& socket){
	boost::asio::ip::tcp::socket::endpoint_type local= socket.local_endpoint();
	boost::asio::ip::tcp::socket::endpoint_type remote= socket.remote_endpoint();

	stringstream ss;
	ss << "{local=" << toString(local) << ", remote=" << toString(remote) << "}";
	return ss.str();
}

std::string util::toString(const boost::asio::ip::tcp::socket::endpoint_type& endpoint){
	stringstream ss;
	ss << endpoint.address() << "/" << endpoint.port();
	return ss.str();
}
