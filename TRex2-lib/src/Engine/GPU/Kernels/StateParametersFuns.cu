//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Daniele Rogora
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

/**
 * The readéValue functions below are used when a leaf of a binary tree is reached, and it returns the value in the type requested by the computation function. 
 * In case the real type of the attribute is different, a cast is performed 
 */
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
int d_readIntValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node node) {
    if (node.isStatic) {
        if (node.valueType==INT) return node.intVal;
        else if (node.valueType==FLOAT) return node.floatVal;
        else if (node.valueType==BOOL) return node.boolVal;
    }
    int refIndex = node.refersTo;
    if ( node.sType != AGG) {
        //are we checking the current state packet?
        ValType type;
        EventInfo myEvt;
        if (refIndex == stateIndex && node.sType==STATE)
        {
            myEvt = stack[id + offset];
        }
        else {
            myEvt = ev->infos[refIndex];
        }
        type = myEvt.attr[node.attrNum].type;
        if (type==INT) return myEvt.attr[node.attrNum].intVal;
        else if (type==FLOAT) return myEvt.attr[node.attrNum].floatVal;
        else if (type==BOOL) return myEvt.attr[node.attrNum].boolVal;
    } else {
        //This refers to a parameter between states: can't handle aggregates here
    }
    return 0; //Should never be reached, but it suppresses GCC warning
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
float d_readFloatValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node node) {
    if (node.isStatic) {
        if (node.valueType==INT) return node.intVal;
        else if (node.valueType==FLOAT) return node.floatVal;
        else if (node.valueType==BOOL) return node.boolVal;
    }
    int refIndex = node.refersTo;
    if ( node.sType != AGG) {
        //are we checking the current state packet?
        ValType type;
        EventInfo myEvt;
        if (refIndex == stateIndex && node.sType==STATE)
        {
            myEvt = stack[id + offset];
        }
        else {
            myEvt = ev->infos[refIndex];
        }
        type = myEvt.attr[node.attrNum].type;
        if (type==INT) return myEvt.attr[node.attrNum].intVal;
        else if (type==FLOAT) return myEvt.attr[node.attrNum].floatVal;
        else if (type==BOOL) return myEvt.attr[node.attrNum].boolVal;
    } else {
        //This refers to a parameter between states: can't handle aggregates here
    }
    return 0; //Should never be reached, but it suppresses GCC warning
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
bool d_readBoolValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node node) {
    if (node.isStatic) {
        if (node.valueType==INT) return node.intVal;
        else if (node.valueType==FLOAT) return node.floatVal;
        else if (node.valueType==BOOL) return node.boolVal;
    }
    int refIndex = node.refersTo;
    if ( node.sType != AGG) {
        //are we checking the current state packet?
        ValType type;
        EventInfo myEvt;
        if (refIndex == stateIndex && node.sType==STATE)
        {
            myEvt = stack[id + offset];
        }
        else {
            myEvt = ev->infos[refIndex];
        }
        type = myEvt.attr[node.attrNum].type;
        if (type==INT) return myEvt.attr[node.attrNum].intVal;
        else if (type==FLOAT) return myEvt.attr[node.attrNum].floatVal;
        else if (type==BOOL) return myEvt.attr[node.attrNum].boolVal;
    } else {
        //This refers to a parameter between states: can't handle aggregates here
    }
    return false; //Should never be reached, but it suppresses GCC warning
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
char *d_readStringValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *node) {
    if (node->isStatic) {
        if (node->valueType==STRING) {
            return node->stringVal;
        }
    }
    int refIndex = node->refersTo;
    ValType type;
    EventInfo *myEvt;
    if (refIndex == stateIndex && node->sType==STATE)
    {
        myEvt = &stack[id + offset];
    }
    else {
        myEvt = &ev->infos[refIndex];
    }
    type = myEvt->attr[node->attrNum].type;
    if (type==STRING) return myEvt->attr[node->attrNum].stringVal;
    return NULL;
}

/**
 * The compute*Value functions below use a simple small stack to parse the serialized version of the binary operational trees 
 */
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
int d_computeIntValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    int stack[STACK_SIZE];
    int stackPtr=0;
    OpTreeOperation op;
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            stack[stackPtr++] = d_readIntValue(id, offset, ev, isNeg, stateIndex, tree[i]);
        }
        else if (tree[i].type==INNER) {
            op = tree[i].operation;
            if (op==ADD) stack[stackPtr++] = stack[--stackPtr] + stack[--stackPtr];
            else if (op==SUB) stack[stackPtr++] = stack[--stackPtr] - stack[--stackPtr];
            else if (op==MUL) stack[stackPtr++] = stack[--stackPtr] * stack[--stackPtr];
            else if (op==DIV) stack[stackPtr++] = stack[--stackPtr] / stack[--stackPtr];
        }
    }
    return stack[0];
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
float d_computeFloatValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    float stack[STACK_SIZE];
    int stackPtr=0;
    OpTreeOperation op;
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            stack[stackPtr++] = d_readFloatValue(id, offset, ev, isNeg, stateIndex, tree[i]);
        }
        else if (tree[i].type==INNER) {
            op = tree[i].operation;
            if (op==ADD) stack[stackPtr++] = stack[--stackPtr] + stack[--stackPtr];
            else if (op==SUB) stack[stackPtr++] = stack[--stackPtr] - stack[--stackPtr];
            else if (op==MUL) stack[stackPtr++] = stack[--stackPtr] * stack[--stackPtr];
            else if (op==DIV) stack[stackPtr++] = stack[--stackPtr] / stack[--stackPtr];
        }
    }
    return stack[0];
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
bool d_computeBoolValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    bool stack[STACK_SIZE];
    int stackPtr=0;
    OpTreeOperation op;
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            stack[stackPtr++] = d_readBoolValue(id, offset, ev, isNeg, stateIndex, tree[i]);
        }
        else if (tree[i].type==INNER) {
            op = tree[i].operation;
            if (op==AND) stack[stackPtr++] = stack[--stackPtr] && stack[--stackPtr];
            else if (op==OR) stack[stackPtr++] = stack[--stackPtr] || stack[--stackPtr];
        }
    }
    return stack[0];
}

/*
 * String values can't be used with binary trees of operations!
 */
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
char *d_computeStringValue(int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            return d_readStringValue(id, offset, ev, isNeg, stateIndex, &tree[i]);
        }
    }
    return NULL;
}