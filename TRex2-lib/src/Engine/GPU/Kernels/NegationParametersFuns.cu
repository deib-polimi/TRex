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


__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD		
__noinline__ 
#endif
int d_readIntValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node node) {
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
        if (node.sType==NEG)
        {
            myEvt = stack[id + offset];
        }
        else if (refIndex == stateIndex && (node.sType==STATE))
        {
            myEvt = *ev1;
        }
        else {
            myEvt = ev->infos[refIndex];
        }
        type = myEvt.attr[node.attrNum].type;
        if (type==INT) return myEvt.attr[node.attrNum].intVal;
        else if (type==FLOAT) return myEvt.attr[node.attrNum].floatVal;
        else if (type==BOOL) return myEvt.attr[node.attrNum].boolVal;
    } else {
    }
    return 0;
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
float d_readFloatValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node node) {
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
        if (node.sType==NEG)
        {
            myEvt = stack[id + offset];
        }
        else if (refIndex == stateIndex && (node.sType==STATE))
        {
            myEvt = *ev1;
        }
        else {
            myEvt = ev->infos[refIndex];
        }
        type = myEvt.attr[node.attrNum].type;
        if (type==INT) return myEvt.attr[node.attrNum].intVal;
        else if (type==FLOAT) return myEvt.attr[node.attrNum].floatVal;
        else if (type==BOOL) return myEvt.attr[node.attrNum].boolVal;
    } else {
    }
    return 0;
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
bool d_readBoolValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node node) {
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
        if (node.sType==NEG)
        {
            myEvt = stack[id + offset];
        }
        else if (refIndex == stateIndex && (node.sType==STATE))
        {
            myEvt = *ev1;
        }
        else {
            myEvt = ev->infos[refIndex];
        }
        type = myEvt.attr[node.attrNum].type;
        if (type==INT) return myEvt.attr[node.attrNum].intVal;
        else if (type==FLOAT) return myEvt.attr[node.attrNum].floatVal;
        else if (type==BOOL) return myEvt.attr[node.attrNum].boolVal;
    } else {
    }
    return false;
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
char *d_readStringValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *node) {
    if (node->isStatic) {
        if (node->valueType==STRING) {
            return node->stringVal;
        }
    }
    int refIndex = node->refersTo;
    if ( node->sType != AGG) {
        //are we checking the current state packet?
        ValType type;
        EventInfo *myEvt;
        if (node->sType==NEG)
        {
            myEvt = &stack[id + offset];
        }
        else if (refIndex == stateIndex && (node->sType==STATE))
        {
            myEvt = ev1;
        }
        else {
            myEvt = &ev->infos[refIndex];
        }
        type = myEvt->attr[node->attrNum].type;
        if (type==STRING) {
            return myEvt->attr[node->attrNum].stringVal;
        }
    }
    return NULL;
}


__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
int d_computeIntValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    int stack[STACK_SIZE];
    int stackPtr=0;
    OpTreeOperation op;
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            stack[stackPtr++] = d_readIntValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, tree[i]);
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
float d_computeFloatValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    float stack[STACK_SIZE];
    int stackPtr=0;
    OpTreeOperation op;
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            stack[stackPtr++] = d_readFloatValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, tree[i]);
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
bool d_computeBoolValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    bool stack[STACK_SIZE];
    int stackPtr=0;
    OpTreeOperation op;
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            stack[stackPtr++] = d_readBoolValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, tree[i]);
        }
        else if (tree[i].type==INNER) {
            op = tree[i].operation;
            if (op==AND) stack[stackPtr++] = stack[--stackPtr] && stack[--stackPtr];
            else if (op==OR) stack[stackPtr++] = stack[--stackPtr] || stack[--stackPtr];
        }
    }
    return stack[0];
}

__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
char *d_computeStringValueForNegation(EventInfo *ev1, int id, int offset, EventInfoSet *ev, bool isNeg, int stateIndex, Node *tree, int size) {
    for (int i=size-1; i>=0; i--) {
        if (tree[i].type==LEAF) {
            //It's a "number"
            return d_readStringValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, &tree[i]);
        }
    }
    return NULL;
}