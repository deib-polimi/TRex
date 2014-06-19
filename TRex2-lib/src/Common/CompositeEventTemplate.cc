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

#include "CompositeEventTemplate.h"

using namespace std;

CompositeEventTemplate::CompositeEventTemplate(int parEventType) {
	eventType = parEventType;
}

CompositeEventTemplate::~CompositeEventTemplate() {
	for (map<int, CompositeEventTemplateAttr *>::iterator it=attributes.begin(); it!=attributes.end(); ++it) {
		delete it->second;
	}
}

void CompositeEventTemplate::addAttribute(char *name, OpTree *value) {
	CompositeEventTemplateAttr *attribute = new CompositeEventTemplateAttr(name, value);
	attributes.insert(make_pair(attributes.size(), attribute));
}

void CompositeEventTemplate::addStaticAttribute(Attribute pattr) {
	CompositeEventTemplateStaticAttr *attribute = new CompositeEventTemplateStaticAttr(pattr);
	staticAttributes.insert(make_pair(staticAttributes.size(), attribute));
}

int CompositeEventTemplate::getEventType() {
	return eventType;
}

int CompositeEventTemplate::getAttributesNum() {
	return attributes.size();
}

int CompositeEventTemplate::getStaticAttributesNum() {
	return staticAttributes.size();
}

void CompositeEventTemplate::getAttributeName(char *name, int index) {
	attributes[index]->getName(name);
}

void CompositeEventTemplate::getStaticAttributeName(char *name, int index) {
	staticAttributes[index]->getName(name);
}

char *CompositeEventTemplate::getAttributeName(int index) {
	return attributes[index]->getName();
}

OpTree * CompositeEventTemplate::getAttributeTree(int index) {
	return attributes[index]->getValue();
}

CompositeEventTemplate * CompositeEventTemplate::dup() {
	CompositeEventTemplate *dup = new CompositeEventTemplate(eventType);
	for (map<int, CompositeEventTemplateAttr *>::iterator it=attributes.begin(); it!=attributes.end(); ++it) {
		CompositeEventTemplateAttr *attrDup = it->second->dup();
		dup->addAttribute(attrDup);
	}
	for (map<int, CompositeEventTemplateStaticAttr *>::iterator it=staticAttributes.begin(); it!=staticAttributes.end(); ++it) {
		CompositeEventTemplateStaticAttr *attrDup = it->second->dup();
		dup->addStaticAttribute(attrDup);
	}
	return dup;
}

void CompositeEventTemplate::addAttribute(CompositeEventTemplateAttr *parAttr) {
	attributes.insert(make_pair(attributes.size(), parAttr));
}

void CompositeEventTemplate::addStaticAttribute(CompositeEventTemplateStaticAttr *parAttr) {
	staticAttributes.insert(make_pair(staticAttributes.size(), parAttr));
}

void CompositeEventTemplate::getStaticAttribute(Attribute& pAtt, int index)
{
	pAtt = staticAttributes[index]->getAttr();
}
