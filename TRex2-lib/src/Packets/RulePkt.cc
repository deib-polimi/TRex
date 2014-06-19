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

#include "RulePkt.h"

using namespace std;

int RulePkt::lastId = 0;

//#if MP_MODE == MP_COPY
/**
 * Creates an exact copy of the packet
 */
RulePkt* RulePkt::copy()
{
    RulePkt *copy = new RulePkt(false);
    copy->addRootPredicate(this->getPredicate(0).eventType, this->getPredicate(0).constraints, this->getPredicate(0).constraintsNum);
    for (int i=1; i<this->getPredicatesNum(); i++) {
        copy->addPredicate(this->getPredicate(i).eventType, this->getPredicate(i).constraints, this->getPredicate(i).constraintsNum, this->getPredicate(i).refersTo, this->getPredicate(i).win, this->getPredicate(i).kind );
    }
    copy->setCompositeEventTemplate(this->getCompositeEventTemplate());
    return copy;
}

//#endif


RulePkt::RulePkt(bool resetCount) {
    if (resetCount) lastId = 0;
    ruleId = lastId++;
    ceTemplate = NULL;
}

RulePkt::~RulePkt() {
    for (map<int, Predicate>::iterator it=predicates.begin(); it!=predicates.end(); ++it) {
        delete it->second.constraints;
    }
    for (map<int, Negation>::iterator it=negations.begin(); it!=negations.end(); ++it) {
        delete it->second.constraints;
    }
    for (map<int, Aggregate>::iterator it=aggregates.begin(); it!=aggregates.end(); ++it) {
        delete it->second.constraints;
    }
    if (ceTemplate!=NULL) delete ceTemplate;
}

bool RulePkt::addRootPredicate(int eventType, Constraint constr[], int constrLen) {
    if (predicates.size()>0) return false;
    Predicate p;
    p.eventType = eventType;
    p.refersTo = -1;
    p.win = 0;
    p.kind = EACH_WITHIN;
    p.constraintsNum = constrLen;
    p.constraints = new Constraint[constrLen];
    for (int i=0; i<constrLen; i++) p.constraints[i] = constr[i];
    predicates.insert(make_pair(predicates.size(), p));
    return true;
}

bool RulePkt::addPredicate(int eventType, Constraint constr[], int constrLen, int refersTo, TimeMs &win, CompKind kind) {
    int numPredicates = predicates.size();
    if (numPredicates<=0 || refersTo>=numPredicates) return false;
    Predicate p;
    p.eventType = eventType;
    p.refersTo = refersTo;
    p.win = win;
    p.kind = kind;
    p.constraintsNum = constrLen;
    p.constraints = new Constraint[constrLen];
    for (int i=0; i<constrLen; i++) p.constraints[i] = constr[i];
    predicates.insert(make_pair(predicates.size(), p));
    return true;
}

bool RulePkt::addTimeBasedNegation(int eventType, Constraint *constraints, int constrLen, int referenceId, TimeMs &win) {
    return addNegation(eventType, constraints, constrLen, -1, win, referenceId);
}

bool RulePkt::addNegationBetweenStates(int eventType, Constraint *constraints, int constrLen, int id1, int id2) {
    TimeMs unused;
    int lowerId, upperId;
    if (id1<id2) {
        upperId=id1;
        lowerId=id2;
    } else {
        upperId=id2;
        lowerId=id1;
    }
    return addNegation(eventType, constraints, constrLen, lowerId, unused, upperId);
}

bool RulePkt::addParameterBetweenStates(int id1, char *name1, int id2, char *name2) {
    int lowerId, upperId;
    char *lowerName;
    char *upperName;
    if (id1<id2) {
        upperId=id1;
        upperName=name1;
        lowerId=id2;
        lowerName=name2;
    } else {
        upperId=id2;
        upperName=name2;
        lowerId=id1;
        lowerName=name1;
    }
    return addParameter(lowerId, lowerName, upperId, upperName, STATE);
}

int RulePkt::findLastState(OpTree* tree)
{
    if (tree->getType() == LEAF) {
        OpValueReference *reference = tree->getValueReference();
        RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
        if (pktReference == NULL) {
            StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
            if (sReference != NULL) {
                return -1;
            }
        }
        else {
            if (pktReference->refersToAgg() || pktReference->refersToNeg()) {
                return -1;
            }
            return pktReference->getIndex();
        }
    }
    else {
        return max(findLastState(tree->getLeftSubtree()), findLastState(tree->getRightSubtree()));
    }
}


int RulePkt::findAggregate(OpTree* tree)
{
    if (tree->getType() == LEAF) {
        OpValueReference *reference = tree->getValueReference();
        RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
        if (pktReference == NULL) {
            StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
            if (sReference != NULL) {
                return -1;
            }
        }
        else {
            if (pktReference->refersToAgg()) {
                return pktReference->getIndex();
            }
            return -1;
        }
    }
    else {
        return max(findAggregate(tree->getLeftSubtree()), findAggregate(tree->getRightSubtree()));
    }
}


int RulePkt::findNegation(OpTree* tree)
{
    if (tree->getType() == LEAF) {
        OpValueReference *reference = tree->getValueReference();
        RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
        if (pktReference == NULL) {
            StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
            if (sReference != NULL) {
                return -1;
            }
        }
        else {
            if (pktReference->refersToNeg()) {
                return pktReference->getIndex();
            }
            return -1;
        }
    }
    else {
        return max(findNegation(tree->getLeftSubtree()), findNegation(tree->getRightSubtree()));
    }
}

bool RulePkt::addComplexParameterForAggregate(Op pOperation, ValType type, OpTree* leftTree, OpTree* rightTree)
{
    CPUParameter p;
    p.operation = pOperation;
    p.leftTree = leftTree;
    p.rightTree = rightTree;
    p.vtype = type;
    p.type = AGG;
    int lIndex = max(findAggregate(leftTree), findAggregate(rightTree));
    if (lIndex < 0) {
        return false;
    }
    p.lastIndex = lIndex;
    complexParameters.insert(make_pair(complexParameters.size(), p));


    int depth = 2;
    GPUParameter gp;
    gp.aggIndex = lIndex;
    //Inversion for the GPU
    gp.lastIndex = -1;
    depth = max(findDepth(leftTree, depth), findDepth(rightTree, depth));
    gp.depth = depth;
    gp.sType = AGG;
    gp.vType = type;
    gp.operation = pOperation;
    serializeTrees(leftTree, rightTree, depth, gp);
    complexGPUParameters.insert(make_pair(complexGPUParameters.size(), gp));
    return true;
}


bool RulePkt::addComplexParameterForNegation(Op pOperation, ValType type, OpTree* leftTree, OpTree* rightTree)
{
    CPUParameter p;
    p.operation = pOperation;
    p.leftTree = leftTree;
    p.rightTree = rightTree;
    p.vtype = type;
    p.type = NEG;
    int lIndex = max(findNegation(leftTree), findNegation(rightTree));
    if (lIndex < 0) {
        return false;
    }
    p.lastIndex = lIndex;
    complexParameters.insert(make_pair(complexParameters.size(), p));


    int depth = 2;
    GPUParameter gp;
    gp.negIndex = lIndex;
    //Inversion for the GPU
    gp.lastIndex = this->getPredicatesNum() -  max(findLastState(leftTree), findLastState(rightTree)) - 1;
    depth = max(findDepth(leftTree, depth), findDepth(rightTree, depth));
    gp.depth = depth;
    gp.sType = NEG;
    gp.vType = type;
    gp.operation = pOperation;
    serializeTrees(leftTree, rightTree, depth, gp);
    complexGPUParameters.insert(make_pair(complexGPUParameters.size(), gp));
    return true;
}

int RulePkt::findDepth(OpTree* tree, int depth)
{
    if (tree->getType() == LEAF) {
        return depth;
    }
    else {
        return max(findDepth(tree->getLeftSubtree(), depth+1), findDepth(tree->getRightSubtree(), depth+1));
    }
}

void RulePkt::serializeTrees(OpTree* ltree, OpTree* rtree, int depth, GPUParameter &gp)
{
    //clear the array
    for (int i=0; i< (1 << depth) - 1; i++) {
        gp.rightTree[i].empty = gp.leftTree[i].empty = true;
        gp.rightTree[i].refersTo = gp.leftTree[i].refersTo = -1;
        gp.rightTree[i].attrNum = gp.leftTree[i].attrNum = -1;
        gp.rightTree[i].intVal = gp.leftTree[i].intVal = -1;
        gp.rightTree[i].type = gp.leftTree[i].type = INNER;
        gp.rightTree[i].sType = gp.leftTree[i].sType = STATE;
        gp.rightTree[i].valueType = gp.leftTree[i].valueType = INT;
    }
    int cnt=0;
    serializeNode(ltree, cnt, gp.leftTree);
    gp.lSize = cnt;
    cnt=0;
    serializeNode(rtree, cnt, gp.rightTree);
    gp.rSize = cnt;
}

void RulePkt::serializeNode(OpTree* tree, int &idx, Node serialized[])
{
    serialized[idx].empty = false;
    if (tree->getType() == LEAF) {
        serialized[idx].type = LEAF;
        OpValueReference *reference = tree->getValueReference();
        RulePktValueReference *pktReference = dynamic_cast<RulePktValueReference *>(reference);
        if (pktReference == NULL) {
            StaticValueReference *sReference = dynamic_cast<StaticValueReference *>(reference);
            if (sReference != NULL) {
                serialized[idx].isStatic = true;
                serialized[idx].valueType = sReference->getType();
                switch (sReference->getType()) {
                case INT:
                    serialized[idx].intVal = sReference->getIntValue();
                    break;

                case FLOAT:
                    serialized[idx].floatVal = sReference->getFloatValue();
                    break;
                }
            }
        }
        else {
            if (pktReference->refersToAgg()) serialized[idx].sType = AGG;
            else if (pktReference->refersToNeg()) serialized[idx].sType = NEG;
            else serialized[idx].sType = STATE;
            serialized[idx].isStatic = false;
            //inversion needed for the GPU
            if (serialized[idx].sType==STATE) {
                serialized[idx].refersTo = this->getPredicatesNum() - pktReference->getIndex() -1;
            } else {
                serialized[idx].refersTo = pktReference->getIndex();
            }
            if (!pktReference->refersToAgg()) {
                strcpy(serialized[idx].attrName, pktReference->getName());
            }
        }
        idx++;
    }
    else {
        serialized[idx].type = INNER;
        serialized[idx].operation = tree->getOp();
        idx++;
        serializeNode(tree->getLeftSubtree(), idx, serialized);
        serializeNode(tree->getRightSubtree(), idx, serialized);
    }
}


bool RulePkt::addComplexParameter(Op pOperation, ValType type, OpTree *leftTree, OpTree *rightTree)
{
    CPUParameter p;
    p.operation = pOperation;
    p.leftTree = leftTree;
    p.rightTree = rightTree;
    p.vtype = type;
    p.type = STATE;
    int lIndex = max(findLastState(leftTree), findLastState(rightTree));
    if (lIndex < 0) {
        return false;
    }
    p.lastIndex = lIndex;
    complexParameters.insert(make_pair(complexParameters.size(), p));

    int depth = 2;
    GPUParameter gp;
    //Inversion for the GPU
    gp.lastIndex = this->getPredicatesNum() - lIndex - 1;
    depth = max(findDepth(leftTree, depth), findDepth(rightTree, depth));
    gp.depth = depth;
    gp.sType = STATE;
    gp.vType = type;
    gp.operation = pOperation;
    serializeTrees(leftTree, rightTree, depth, gp);
    complexGPUParameters.insert(make_pair(complexGPUParameters.size(), gp));
    return true;
}

bool RulePkt::addParameterForNegation(int id, char *name, int negId, char *negName) {
    return addParameter(id, name, negId, negName, NEG);
}

bool RulePkt::addParameterForAggregate(int id, char *name, int aggId, char *aggName) {
    return addParameter(id, name, aggId, aggName, AGG);
}

bool RulePkt::addTimeBasedAggregate(int eventType, Constraint *constraints, int constrLen, int referenceId, TimeMs &win, char *name, AggregateFun fun) {
    return addAggregate(eventType, constraints, constrLen, -1, win, referenceId, name, fun);
}

bool RulePkt::addAggregateBetweenStates(int eventType, Constraint *constraints, int constrLen, int id1, int id2, char *name, AggregateFun fun) {
    TimeMs unused;
    int lowerId, upperId;
    if (id1<id2) {
        upperId=id1;
        lowerId=id2;
    } else {
        upperId=id2;
        lowerId=id1;
    }
    return addAggregate(eventType, constraints, constrLen, lowerId, unused, upperId, name, fun);
}

bool RulePkt::addConsuming(int eventIndex) {
    int numPredicates = predicates.size();
    if (eventIndex<0 || eventIndex>=numPredicates) return false;
    consuming.insert(eventIndex);
    return true;
}

void RulePkt::getLeaves(set<int> &leaves) {
    map<int, int> referenceCount;
    getReferenceCount(referenceCount);
    for (map<int, int>::iterator it=referenceCount.begin(); it!=referenceCount.end(); it++) {
        if (it->second == 0) leaves.insert(it->first);
    }
}

void RulePkt::getJoinPoints(set<int> &joinPoints) {
    map<int, int> referenceCount;
    getReferenceCount(referenceCount);
    for (map<int, int>::iterator it=referenceCount.begin(); it!=referenceCount.end(); it++)
        if (it->second > 0) joinPoints.insert(it->first);
}

bool RulePkt::containsEventType(int eventType, bool includeNegations) {
    for (map<int, Predicate>::iterator it=predicates.begin(); it!=predicates.end(); ++it) {
        if (it->second.eventType==eventType) return true;
    }
    if (includeNegations) {
        for (map<int, Negation>::iterator it=negations.begin(); it!=negations.end(); ++it) {
            if (it->second.eventType==eventType) return true;
        }
    }
    return false;
}

void RulePkt::getContainedEventTypes(set<int> &evTypes) {
    for (map<int, Predicate>::iterator it=predicates.begin(); it!=predicates.end(); ++it) {
        evTypes.insert(it->second.eventType);
    }
}

TimeMs RulePkt::getMaxWin() {
    set<int> leaves;
    getLeaves(leaves);
    TimeMs returnTime = 0;
    for (set<int>::iterator it=leaves.begin(); it!=leaves.end(); it++) {
        int leaf = *it;
        TimeMs currentTime = getWinBetween(0, leaf);
        if (currentTime > returnTime) returnTime = currentTime;
    }
    return returnTime;
}

TimeMs RulePkt::getWinBetween(int lowerId, int upperId) {
    int currentIndex = upperId;
    TimeMs timeBetween = 0;
    while (currentIndex!=lowerId) {
        timeBetween += predicates[currentIndex].win;
        currentIndex = predicates[currentIndex].refersTo;
    }
    return timeBetween;
}

bool RulePkt::isDirectlyConnected(int id1, int id2) {
    int numPredicates = predicates.size();
    if (id1==id2 || id1<0 || id2<0 || id1>numPredicates || id2>numPredicates) return false;
    if (predicates[id2].refersTo==id1 || predicates[id1].refersTo==id2) return true;
    return false;
}

bool RulePkt::isInTheSameSequence(int id1, int id2) {
    int numPredicates = predicates.size();
    if (id1==id2 || id1<0 || id2<0 || id1>numPredicates || id2>numPredicates) return false;
    int min, max;
    if (id1<id2) {
        min=id1;
        max=id2;
    } else {
        min=id2;
        max=id1;
    }
    int i=max;
    while(i>0) {
        i=predicates[i].refersTo;
        if (i==min) return true;
    }
    return false;
}

bool RulePkt::operator<(const RulePkt &pkt) const {
    return ruleId<pkt.ruleId;
}

bool RulePkt::operator==(const RulePkt &pkt) const {
    return ruleId==pkt.ruleId;
}

bool RulePkt::operator!=(const RulePkt &pkt) const {
    return ruleId!=pkt.ruleId;
}

bool RulePkt::addNegation(int eventType, Constraint *constr, int constrLen, int lowerId, TimeMs &lowerTime, int upperId) {
    int numPredicates = predicates.size();
    if (lowerId>numPredicates) return false;
    if (upperId<0 || upperId>numPredicates) return false;
    if (lowerId>=0 && lowerId<=upperId) return false;
    Negation n;
    n.eventType = eventType;
    n.lowerId = lowerId;
    if (lowerId<0) n.lowerTime = lowerTime;
    n.upperId = upperId;
    n.constraintsNum = constrLen;
    n.constraints = new Constraint[constrLen];
    for (int i=0; i<constrLen; i++) n.constraints[i] = constr[i];
    negations.insert(make_pair(negations.size(), n));
    return true;
}

bool RulePkt::addParameter(int index1, char *name1, int index2, char *name2, StateType type) {
    int numPredicates = predicates.size();
    int numAggregates = aggregates.size();
    int numNegations = negations.size();
    if (index1<0 || index1>numPredicates) return false;
    if (index2<0) return false;
    if (type==STATE && index2>numPredicates) return false;
    if (type==AGG && index2>numAggregates) return false;
    if (type==NEG && index2>numNegations) return false;
    Parameter p;
    p.evIndex1 = index1;
    p.evIndex2 = index2;
    p.type = type;
    strcpy(p.name1, name1);
    strcpy(p.name2, name2);
    parameters.insert(make_pair(parameters.size(), p));
    return true;
}

bool RulePkt::addAggregate(int eventType, Constraint *constr, int constrLen, int lowerId, TimeMs &lowerTime, int upperId, char *name, AggregateFun fun) {
    int numPredicates = predicates.size();
    if (lowerId>numPredicates) return false;
    if (upperId<0 || upperId>numPredicates) return false;
    if (lowerId>=0 && lowerId<=upperId) return false;
    Aggregate a;
    a.eventType = eventType;
    a.lowerId = lowerId;
    if (lowerId<0) a.lowerTime = lowerTime;
    a.upperId = upperId;
    a.constraintsNum = constrLen;
    a.constraints = new Constraint[constrLen];
    for (int i=0; i<constrLen; i++) a.constraints[i] = constr[i];
    strcpy(a.name, name);
    a.fun = fun;
    aggregates.insert(make_pair(aggregates.size(), a));
    return true;
}

void RulePkt::getReferenceCount(map<int, int> &referenceCount) {
    int numPredicates = predicates.size();
    for (int i=0; i<numPredicates; i++) referenceCount.insert(make_pair(i, 0));
    for (int i=1; i<numPredicates; i++) {
        int referredIndex = predicates[i].refersTo;
        map<int, int>::iterator it = referenceCount.find(referredIndex);
        int count = it->second + 1;
        referenceCount.erase(it);
        referenceCount.insert(make_pair(referredIndex, count));
    }
}
