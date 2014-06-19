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

#ifndef OPTREE_H_
#define OPTREE_H_

#include <stdlib.h>
#include "OpValueReference.h"
#include "../Common/Consts.h"

/**
 * Represents the type of a tree: it can be a leaf (a value), or an inner node (a subtree)
 */
enum OpTreeType {
	LEAF=0,
	INNER=1
};

/**
 * Possible operations stored at inner nodes
 */
enum OpTreeOperation {
	ADD=0,
	SUB=1,
	MUL=2,
	DIV=3,
	AND=4,
	OR=5
};

/**
 * An OpTree represents a mathematical operation using a binary tree.
 * It is used to compute the value of an attribute of a composite events
 * starting from the values of the attributes of simple events generating it.
 * An OpTree can be built by recursively creating its composing subtrees.
 * Actual values are stored at leaves: to make the data structure generic enough,
 * values are not referenced directly, but through an OpValueReference class.
 * By re-defining OpValueReference through sub-classes, it is possible to
 * store references to different kinds of data structures.
 */
class OpTree {
public:

	/**
	 * Leaf node constructor
	 */
	OpTree(OpValueReference *value, ValType parValType);

	/**
	 * Builds up an OpTree starting from its left and right subtrees
	 */
	OpTree(OpTree *leftTree, OpTree *rightTree, OpTreeOperation parOp, ValType parValType);

	/**
	 * Destructor
	 */
	virtual ~OpTree();

	/**
	 * Returns the type of the tree
	 */
	OpTreeType getType();

	/**
	 * Returns the type of the value represented by the tree
	 */
	ValType getValType();

	/**
	 * Returns the left subtree
	 */
	OpTree * getLeftSubtree();

	/**
	 * Returns the right subtree
	 */
	OpTree * getRightSubtree();

	/**
	 * Returns the operation represented by the tree.
	 * Requires the tree not to be a leaf.
	 */
	OpTreeOperation getOp();

	/**
	 * Returns the reference to the actual value.
	 * Requires the tree to be a leaf.
	 */
	OpValueReference * getValueReference();

	/**
	 * Set the reference to the actual value.
	 * Requires the tree to be a leaf.
	 * Before setting the new valueReference, it deletes the old one.
	 */
	void changeValueReference(OpValueReference *value);

	/**
	 * Creates an exacy copy (deep copy) of the data structure
	 */
	OpTree * dup();

private:
	OpTreeType type;					// Type of the tree
	OpTree *leftTree;					// Left subtree
	OpTree *rightTree;				// Right subtree
	OpValueReference *value;	// Reference to the actual value (in case of leaf type)
	OpTreeOperation op;				// Operation to be performed (in case of inner type)
	ValType valType;					// Type of the value generated or referenced
};

#endif /* OPTREE_H_ */
