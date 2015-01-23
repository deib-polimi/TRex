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
 * All the methods to compute values for each differenet ValType and for each type of computation have been splitted and placed in differenet files according to the kind of computation that they belong to.
 *
 */
#include "StateParametersFuns.cu"
#include "AggregateParametersFuns.cu"
#include "NegationParametersFuns.cu"
#include "../../../Common/Consts.h"

//Returns true if matches, that is expression verified
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
bool d_checkComplexParameter(int id, int offset, GPUParameter *par, EventInfoSet *ev, bool isNeg, int stateIndex) {
    //Analyze left tree, right tree and compare
    Op operation = par->operation;
    ValType vType = par->vType;
    if (vType==INT) {
        int left = d_computeIntValue(id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        int right = d_computeIntValue(id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==GT) return left>right;
        if (operation==LT) return left<right;
        if (operation==NE) return left!=right;
    }
    else if (vType==FLOAT) {
        float left = d_computeFloatValue(id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        float right = d_computeFloatValue(id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==GT) return left>right;
        if (operation==LT) return left<right;
        if (operation==NE) return left!=right;
    }
    else if (vType==BOOL) {
        bool left = d_computeBoolValue(id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        bool right = d_computeBoolValue(id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return left == right;
        if (operation==NE) return left != right;
    }
    else if (vType==STRING) {
        char *left, *right;
        left = d_computeStringValue(id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        right = d_computeStringValue(id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return (d_strcmp(left, right)==0);
        if (operation==NE) return (d_strcmp(left, right)!=0);
    }
    return false;
}

//Returns true if matches, that is expression verified
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
bool d_checkComplexParameterForNegation(EventInfo *ev1, int id, int offset, GPUParameter *par, EventInfoSet *ev, bool isNeg, int stateIndex) {
    //Analyze left tree, right tree and compare
    Op operation = par->operation;
    ValType vType = par->vType;

    if (vType==INT) {
        int left = d_computeIntValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        int right = d_computeIntValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==GT) return left>right;
        if (operation==LT) return left<right;
        if (operation==NE) return left!=right;
    }
    else if (vType==FLOAT) {
        float left = d_computeFloatValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        float right = d_computeFloatValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==GT) return left>right;
        if (operation==LT) return left<right;
        if (operation==NE) return left!=right;
    }
    else if (vType==BOOL) {
        bool left = d_computeBoolValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        bool right = d_computeBoolValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==NE) return left!=right;
    }
    else if (vType==STRING) {
        char *left, *right;
        left = d_computeStringValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->leftTree, par->lSize);
        right = d_computeStringValueForNegation(ev1, id, offset, ev, isNeg, stateIndex, par->rightTree, par->rSize);
        if (operation==EQ) return (d_strcmp(left, right)==0);
        if (operation==NE) return (d_strcmp(left, right)!=0);
    }
    return false;
}

//Returns true if matches, that is expression verified
__device__ 
#if NAME_LEN + STRING_LEN > INLINE_THRESHOLD
__noinline__ 
#endif
bool d_checkComplexParameterForAggregate(int id, int offset, GPUParameter *par, EventInfoSet *ev) {
    //Analyze left tree, right tree and compare
    Op operation = par->operation;
    ValType vType = par->vType;

    if (vType==INT) {
        int left = d_computeIntValueForAggregate(id, offset, ev, par->leftTree, par->lSize);
        int right = d_computeIntValueForAggregate(id, offset, ev, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==GT) return left>right;
        if (operation==LT) return left<right;
        if (operation==NE) return left!=right;
    }
    else if (vType==FLOAT) {
        float left = d_computeFloatValueForAggregate(id, offset, ev, par->leftTree, par->lSize);
        float right = d_computeFloatValueForAggregate(id, offset, ev, par->rightTree, par->rSize);
        if (operation==EQ) return left==right;
        if (operation==GT) return left>right;
        if (operation==LT) return left<right;
        if (operation==NE) return left!=right;
    }
    else if (vType==BOOL) {
        bool left = d_computeBoolValueForAggregate(id, offset, ev, par->leftTree, par->lSize);
        bool right = d_computeBoolValueForAggregate(id, offset, ev, par->rightTree, par->rSize);
        if (operation==EQ) return left == right;
        if (operation==NE) return left != right;
    }
    else if (vType==STRING) {
        char *left, *right;
        left = d_computeStringValueForAggregate(id, offset, ev, par->leftTree, par->lSize);
        right = d_computeStringValueForAggregate(id, offset, ev, par->rightTree, par->rSize);
        if (operation==EQ) return (d_strcmp(left, right)==0);
        if (operation==NE) return (d_strcmp(left, right)!=0);
    }
    return false;
}

