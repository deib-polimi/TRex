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


#include "MemoryManager.h"


using namespace std;


MemoryManager::~MemoryManager()
{
  int e=0;
     for (int i=0; i<MAX_CONCURRENT_RULES; i++) {
        e += cudaFree((workspaces[i].prevResultsPtr));
        e += cudaFree((workspaces[i].currentResultsPtr));
	e += cudaFree((workspaces[i].seqMaxPtr));
	e += cudaFree((workspaces[i].alivePtr));
	e += cudaFree((workspaces[i].aggregatesValPtr));
    }

    for (int i=0; i<MAX_RULE_NUM; i++) {
        e += cudaFree((rules[i].resPtr));
        e += cudaFree((rules[i].resultsSizePtr));
        e += cudaFree((rules[i].parametersPtr));
        e += cudaFree((rules[i].aggregatesParametersPtr));
        e += cudaFree((rules[i].negationsParametersPtr));
        e += cudaFree((rules[i].negationsPtr));
    }
    cudaFree(base_address);
    free(chunks);
    if (e) {
	cout << "There was an error freeing GPU memory!" << endl;
    }
}

void MemoryManager::getPageIndexes(int id, int state, passer &res) {
    for (int i=0; i<rules[id].numEventChunks[state]; i++) {
        res.pageIdx[i] = chunks[rules[id].eventsChunks[state][i]].pageIdx % d_chunksnum;
    }
}

void MemoryManager::getNegationsIndexes(int id, int state, passer &res) {
    for (int i=0; i<rules[id].numNegationsChunks[state]; i++) {
        res.pageIdx[i] = chunks[rules[id].negationsChunks[state][i]].pageIdx % d_chunksnum;
    }
}

void MemoryManager::getAllNegationsIndexes(int id, int index, int nNum, negsPasser &res) {
    for (int i=0; i<rules[id].numNegationsChunks[nNum]; i++) {
        res.pages[index][i] = chunks[rules[id].negationsChunks[nNum][i]].pageIdx % d_chunksnum;
    }
}

void MemoryManager::getAggregatesIndexes(int id, int state, passer &res) {
    for (int i=0; i<rules[id].numAggregatesChunks[state]; i++) {
        res.pageIdx[i] = chunks[rules[id].aggregatesChunks[state][i]].pageIdx % d_chunksnum;
    }
}

void MemoryManager::getHostAggregatesIndexes(int id, int state, passer &res) {
    for (int i=0; i<rules[id].numAggregatesChunks[state]; i++) {
        res.pageIdx[i] = chunks[rules[id].aggregatesChunks[state][i]].pageIdx;
    }
}

int MemoryManager::subscribe(int seqLen, int aggrNum, int negNum, int paramNum)
{
    int i=0;
    while (i<MAX_RULE_NUM && inUse[i]==true) i++;
    if (i>=MAX_RULE_NUM -1 && inUse[i]==true) {
        cout << "Max rules num reached" << endl;
        return -1;
    }
    for (int j=0; j<seqLen-1; j++) {
        rules[i].eventsChunks[j][0] = getFreeChunk(i);
        chunks[rules[i].eventsChunks[j][0]].onGpu = i;
        rules[i].numEventChunks[j]=1;
    }
    for (int j=0; j<negNum; j++) {
        rules[i].negationsChunks[j][0] = getFreeChunk(i);
        chunks[rules[i].negationsChunks[j][0]].onGpu = i;
        rules[i].numNegationsChunks[j]=1;
    }
    for (int j=0; j<aggrNum; j++) {
        rules[i].aggregatesChunks[j][0] = getFreeChunk(i);
        chunks[rules[i].aggregatesChunks[j][0]].onGpu = i;
        rules[i].numAggregatesChunks[j]=1;
    }
    inUse[i]=true;
    return i;
}

bool MemoryManager::ruleOwns(int id, int page) {
    for (int i=0; i<MAX_RULE_FIELDS-1; i++) {
        for (int j=0; j<MAX_PAGES_PER_STACK; j++) {
            if (rules[id].eventsChunks[i][j]>=0 && chunks[rules[id].eventsChunks[i][j]].pageIdx % d_chunksnum == page % d_chunksnum) return true;
        }
    }
    for (int i=0; i<MAX_NUM_AGGREGATES; i++) {
        for (int j=0; j<MAX_PAGES_PER_STACK; j++) {
            if (rules[id].aggregatesChunks[i][j]>=0 && chunks[rules[id].aggregatesChunks[i][j]].pageIdx % d_chunksnum == page % d_chunksnum) return true;
        }
    }
    for (int i=0; i<MAX_NEGATIONS_NUM; i++) {
        for (int j=0; j<MAX_PAGES_PER_STACK; j++) {
            if (rules[id].negationsChunks[i][j]>=0 && chunks[rules[id].negationsChunks[i][j]].pageIdx % d_chunksnum == page % d_chunksnum) return true;
        }
    }
    return false;
}

int MemoryManager::getFreeChunk(int id) {
    int i=0;
    while (i<chunksnum-1 && (chunks[i].inUse > 0 || ruleOwns(id,i))) i++;
    if (i>=chunksnum -1 && (chunks[i].inUse > 0 || ruleOwns(id,i))) {
    }
    else {
        if (!notified && i >= d_chunksnum) {
            notified = true;
            cout << "Swapping. Assign more GPU memory to improve performance" << endl;
        }
        chunks[i].inUse++;
        return i;

    }
    cout << "Not enough memory to compute!" << endl;
    exit(-1);
}

void MemoryManager::addEventToStack(int id, EventInfo ev, int stack, int index, StateType type) {
    int neededchunks=index/PAGE_SIZE+1;
    int offset, index2, hostOffset, usingRule;
    int newChunk;
//FIXME: i'm adding 1 evt, can't add more than 1 page!
    if (type==STATE) {
        for (int n=neededchunks; n > rules[id].numEventChunks[stack]; n--) {
            newChunk = getFreeChunk(id);
            rules[id].eventsChunks[stack][index/PAGE_SIZE] = newChunk;
            chunks[newChunk % d_chunksnum].onGpu = id;
            rules[id].numEventChunks[stack]++;
        }
        offset = (chunks[rules[id].eventsChunks[stack][index/PAGE_SIZE]].pageIdx % d_chunksnum) * PAGE_SIZE;
        hostOffset = chunks[rules[id].eventsChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
        usingRule = chunks[rules[id].eventsChunks[stack][index/PAGE_SIZE] % d_chunksnum].onGpu;

    }
    else if (type==AGG) {
        for (int n=neededchunks; n > rules[id].numAggregatesChunks[stack]; n--) {
            newChunk = getFreeChunk(id);
            rules[id].aggregatesChunks[stack][index/PAGE_SIZE] = newChunk;
            chunks[newChunk % d_chunksnum].onGpu = id;
            rules[id].numAggregatesChunks[stack]++;
        }
        offset = (chunks[rules[id].aggregatesChunks[stack][index/PAGE_SIZE]].pageIdx % d_chunksnum) * PAGE_SIZE;
        hostOffset = chunks[rules[id].aggregatesChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
        usingRule = chunks[rules[id].aggregatesChunks[stack][index/PAGE_SIZE] % d_chunksnum].onGpu;

    }
    else if (type==NEG) {
        for (int n=neededchunks; n > rules[id].numNegationsChunks[stack]; n--) {
            newChunk = getFreeChunk(id);
            rules[id].negationsChunks[stack][index/PAGE_SIZE] = newChunk;
            chunks[newChunk % d_chunksnum].onGpu = id;
            rules[id].numNegationsChunks[stack]++;
        }
        offset = (chunks[rules[id].negationsChunks[stack][index/PAGE_SIZE]].pageIdx % d_chunksnum) * PAGE_SIZE;
        hostOffset = chunks[rules[id].negationsChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
        usingRule = chunks[rules[id].negationsChunks[stack][index/PAGE_SIZE] % d_chunksnum].onGpu;
    }
    index2 = index % PAGE_SIZE;
    h_base_address[hostOffset + index2] = ev;
    if (usingRule == id) {
        cudaMemcpyAsync(base_address + offset + index2, h_base_address + hostOffset + index2, (size_t) sizeof(EventInfo), cudaMemcpyHostToDevice);
    }
}

EventInfo MemoryManager::getEventFromStack(int id, int stack, int index, StateType type) {
    int index2, hostOffset;
    if (type==STATE) {
        hostOffset = chunks[rules[id].eventsChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
    }
    else if (type==AGG) {
        hostOffset = chunks[rules[id].aggregatesChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
    }
    else if (type==NEG) {
        hostOffset = chunks[rules[id].negationsChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
    }
    index2 = index % PAGE_SIZE;
    return h_base_address[hostOffset + index2];
}

EventInfo MemoryManager::getEvent(int id, int stack, int index) {
    int offset = chunks[rules[id].eventsChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
    index %= PAGE_SIZE;
    return h_base_address[offset + index];
}

void MemoryManager::setEvent(int id, int stack, int index, EventInfo ev) {
    int offset = chunks[rules[id].eventsChunks[stack][index/PAGE_SIZE]].pageIdx * PAGE_SIZE;
    index %= PAGE_SIZE;
    h_base_address[offset + index] = ev;
}

void MemoryManager::cleanEventStack(int id, int stack, int idx, int &highIndex) {
    int i=0;
    while ((idx+i+1) % ALLOC_SIZE != highIndex) {
        invalidatePage(id, getPage(id, stack, idx+i, 0, STATE));
        setEvent(id, stack, (idx+i)% ALLOC_SIZE, getEvent(id, stack, (idx+i+1)% ALLOC_SIZE));
        i++;
    }
    highIndex-= 1;
    if (highIndex<0) highIndex = ALLOC_SIZE -1;
}

inline uint32_t MemoryManager::getPage(int id, int s, int offset, int minId, StateType stype) {
    uint32_t pad;
    switch (stype) {
    case STATE:
        pad = chunks[rules[id].eventsChunks[s][offset/PAGE_SIZE]].pageIdx;
        break;

    case NEG:
        pad = chunks[rules[id].negationsChunks[s][offset/PAGE_SIZE]].pageIdx;
        break;

    case AGG:
        pad = chunks[rules[id].aggregatesChunks[s][offset/PAGE_SIZE]].pageIdx;
        break;
    }
    return pad;
}

void MemoryManager::copyPageToDevice(int id, int page) {
    uint32_t d_pad, h_pad;
    d_pad = (page % d_chunksnum) * PAGE_SIZE;
    h_pad = page * PAGE_SIZE;
    cudaMemcpyAsync(base_address + d_pad, h_base_address + h_pad, sizeof(EventInfo)*PAGE_SIZE, cudaMemcpyHostToDevice);
    chunks[page%d_chunksnum].onGpu = id;
}

inline void MemoryManager::invalidatePage(int id, int page) {
    chunks[page%d_chunksnum].onGpu  = -1;
}

inline bool MemoryManager::isOnGpu(int id, int page) {
    return chunks[page%d_chunksnum].onGpu  == id;
}

void MemoryManager::setEventsStacks(int id, int s, int minId, int maxId, StateType stype) {
    int e = 0;
    int offset = 0;
    uint32_t page;
    if (maxId >= minId) {
        while (offset + PAGE_SIZE <= minId) offset += PAGE_SIZE;
        page = getPage(id, s, offset, minId, stype);
        if (!isOnGpu(id, page)) copyPageToDevice(id, page); /*else cout << page << " Was on GPU already for rule " << id << endl;*/
        offset += PAGE_SIZE;

        while (offset + PAGE_SIZE < maxId) {
            page = getPage(id, s, offset, 0, stype);
            if (!isOnGpu(id, page)) copyPageToDevice(id, page);
            offset += PAGE_SIZE;
        };
        if (offset < maxId) {
            page = getPage(id, s, offset, 0, stype);
            if (!isOnGpu(id, page)) copyPageToDevice(id, page);
        }
    }
    else {
        //maxId < minId
        ////////////////////////////  From 0 to maxId //////////////////////////////
        while (offset + PAGE_SIZE < maxId) {
            page = getPage(id, s, offset, 0, stype);
            if (!isOnGpu(id, page)) copyPageToDevice(id, page);
            offset += PAGE_SIZE;
        };
        if (offset < maxId) {
            page = getPage(id, s, offset, 0, stype);
            if (!isOnGpu(id, page)) copyPageToDevice(id, page);
        }
        //////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////  From minId to END_OF_MINID_PAGE //////////////////////////////
        while (offset + PAGE_SIZE <= minId) offset += PAGE_SIZE;
        page = getPage(id, s, offset, minId, stype);
        if (!isOnGpu(id, page)) copyPageToDevice(id, page);
        offset += PAGE_SIZE;
        //////////////////////////////////////////////////////////////////////////////////////////

        ////////////////////////////  From END_OF_MINID_PAGE to ALLOC_SIZE //////////////////////////////
        while (offset < ALLOC_SIZE) {
            page = getPage(id, s, offset, 0, stype);
            if (!isOnGpu(id, page)) copyPageToDevice(id, page);
            offset += PAGE_SIZE;
        }
        //////////////////////////////////////////////////////////////////////////////////////////

    }
    if (e > 0) {
        cout << "Error " << e << " while copying events to the device" << endl;
        exit(-1);
    }
}


void MemoryManager::unsubscribe(int id, int seqLen, int negsNum, int aggrNum) {
    swapped[id]=false;
    inUse[id] = false;
    for (int state=0; state < seqLen-1; state++) {
        for (int i=0; i<MAX_PAGES_PER_STACK; i++) {
	  if (rules[id].eventsChunks[state][i]>=0) {
            chunks[rules[id].eventsChunks[state][i]].inUse--;
            chunks[rules[id].eventsChunks[state][i] % d_chunksnum].onGpu = -1;
	  }
        }
    }

    for (int state=0; state < negsNum; state++) {
        for (int i=0; i<MAX_PAGES_PER_STACK; i++) {
	  if (rules[id].negationsChunks[state][i]>=0) {
            chunks[rules[id].negationsChunks[state][i]].inUse--;
            chunks[rules[id].negationsChunks[state][i] % d_chunksnum ].onGpu = -1;
	  }
        }
    }

    for (int state=0; state < aggrNum; state++) {
        for (int i=0; i<MAX_PAGES_PER_STACK; i++) {
	  if (rules[id].aggregatesChunks[state][i]>=0) {
            chunks[rules[id].aggregatesChunks[state][i]].inUse--;
            chunks[rules[id].aggregatesChunks[state][i] % d_chunksnum].onGpu = -1;
	  }
        }
    }
}

void MemoryManager::swapPointers(int id) {
    swapped[id] = !swapped[id];
}

void MemoryManager::restorePointers(int id) {
    swapped[id] = false;
}

int *MemoryManager::getResPtr(int id) {
    return rules[id].resPtr;
}

uint8_t *MemoryManager::getAlivePtr(int id) {
    return workspaces[rules[id].workspaceNum].alivePtr;
}

int *MemoryManager::getSeqMaxPtr(int id) {
    return workspaces[rules[id].workspaceNum].seqMaxPtr;
}

int *MemoryManager::getResultsSizePtr(int id) {
    return rules[id].resultsSizePtr;
}

EventInfoSet *MemoryManager::getPrevResultsPtr(int id) {
    if (!swapped[id]) return workspaces[rules[id].workspaceNum].prevResultsPtr;
    else return workspaces[rules[id].workspaceNum].currentResultsPtr;
}

float *MemoryManager::getAggregatesValPtr(int id) {
    return workspaces[rules[id].workspaceNum].aggregatesValPtr;
}

EventInfoSet *MemoryManager::getCurrentResultsPtr(int id) {
    if (!swapped[id]) return workspaces[rules[id].workspaceNum].currentResultsPtr;
    else return workspaces[rules[id].workspaceNum].prevResultsPtr;
}

GPUParameter *MemoryManager::getParameterPtr(int id) {
    return rules[id].parametersPtr;
}

GPUParameter *MemoryManager::getAggregateParameterPtr(int id) {
    return rules[id].aggregatesParametersPtr;
}

GPUParameter *MemoryManager::getNegationsParameterPtr(int id) {
    return rules[id].negationsParametersPtr;
}

GPUNegation *MemoryManager::getNegationsPtr(int id) {
    return rules[id].negationsPtr;
}
