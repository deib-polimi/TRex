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

#include "OpTree.h"

OpTree::OpTree(OpValueReference *parValue, ValType parValType) {
	type = LEAF;
	leftTree = NULL;
	rightTree = NULL;
	value = parValue;
	valType = parValType;
}

OpTree::OpTree(OpTree *parLeftTree, OpTree *parRightTree, OpTreeOperation parOp, ValType parValType) {
	type = INNER;
	leftTree = parLeftTree;
	rightTree = parRightTree;
	value = NULL;
	op = parOp;
	valType = parValType;
}

OpTree::~OpTree() {
	if (leftTree!=NULL) delete leftTree;
	if (rightTree!=NULL) delete rightTree;
	if (value!=NULL) delete value;
}

OpTreeType OpTree::getType() {
	return type;
}

ValType OpTree::getValType() {
	return valType;
}

OpTree * OpTree::getLeftSubtree() {
	return leftTree;
}

OpTree * OpTree::getRightSubtree() {
	return rightTree;
}

OpTreeOperation OpTree::getOp() {
	return op;
}

OpValueReference * OpTree::getValueReference() {
	return value;
}

void OpTree::changeValueReference(OpValueReference *parValue) {
	if (type==INNER) return;
	delete value;
	value = parValue;
}

OpTree * OpTree::dup() {
	if (type==LEAF) return new OpTree(value->dup(), valType);
	else return new OpTree(leftTree->dup(), rightTree->dup(), op, valType);
}
