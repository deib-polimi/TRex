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

#ifndef CONSTS_H_
#define CONSTS_H_

#include "TimeMs.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


/**
 * LOG defines whether debugging information have to be printed during exeucution or not
 * LOG = 1 prints debugging information
 * LOG = 0 does not print debugging information
 */
#define LOG 0

/**
 * This defines the maximum NAME_LEN + STRING_VAL_LEN for which all the optimizations of nvcc are enabled, in order to avoid compilation problems.
 * 
 */
#define INLINE_THRESHOLD 10

/**
 * The maximum len allowed for a name of an attribute.
 * It is better to keep this value as low as minimum to enhance the GPU engine's performance
 */
#define NAME_LEN 15

/**
 * The maximum len allowed for a string attribute.
 * It is better to keep this value as low as minimum to enhance the GPU engine's performance
 */
#define STRING_VAL_LEN 15

/**
 * Max recurion depth
 */
#define MAX_RECURSION_DEPTH 10

/**
 * The maximum depth handled by the GPU for binary tree of operations. Min should be 2 (if there's not any inner node
 */
#define MAX_DEPTH 5

/*
 * The maximum number of rules that can be handled concurrently by the GPU engine
 */
#define MAX_RULE_NUM 100

/*
 * The size of memory that will be used by TRex; this will be immediately allocated during the initialization
 */
#define MAX_SIZE 250

/*
 * Size in terms of events of a single page of memory
 */
#define PAGE_SIZE 2048

/*
 * Host memory multiplier for the swapper
 */
#define HOST_MEM_MUL 1

/*
 * Max theoretical number of events that a single column could hold
 */
#define ALLOC_SIZE 65536

/**
 * Max number of new complex events that can be created from a single terminator; it matters only when EACH_WITHIN is used
 */
#define MAX_GEN 100

/**
 * Maximum number of primary events that can be checked when building new complex events
 */
#define MAX_NEW_EVENTS 65536

/*
 * Maximum number of rules that can be concurrently stored on the GPU
 */
#define MAX_CONCURRENT_RULES 1

/*
 * Maximum length in terms of predicates of a single rule
 */
#define MAX_RULE_FIELDS 5

/**
 * Maximum number of aggregates per rule
 */
#define MAX_NUM_AGGREGATES 3

/**
 * Maximum number of attributes for events handled by the GPU
 */
#define MAX_NUM_ATTR 5

/**
 * Maximum number of negations
 */
#define MAX_NEGATIONS_NUM 2

/*
 * Maximum number of parameters for {states/aggregates/negations}
 */
#define MAX_PARAMETERS_NUM 3

#define TREE_SIZE (1 << MAX_DEPTH) - 1
#define STACK_SIZE (1 << (MAX_DEPTH - 1))
#define MAX_PAGES_PER_STACK ALLOC_SIZE / PAGE_SIZE

#define BIG_NUM 1048576
#define SMALL_NUM -1048576

#define RUN_GPU true

#define GPU_THREADS 256

//#define CHECK_NEG_PARALLEL

/**
 * MP_MODE defines how multithreading is implemented.
 * MP_MODE = MP_COPY makes the program copy packets to be processed before passing them to the different processing thread.
 * MP_MODE = MP_LOCK allows all thread to share a common copy of the packets to be processed. Locking is used for mutual exclusion.
 */

#define MP_COPY 0
#define MP_LOCK 1
#define MP_MODE MP_LOCK


/**
 * Kinds of packets
 */
enum PktType {
	PUB_PKT=0,
	RULE_PKT=1,
	SUB_PKT=2,
	ADV_PKT=3,
	JOIN_PKT=4
};

/**
 * Kinds of types for the values of attributes and contraints
 */
enum ValType {
	INT=0,
	FLOAT=1,
	BOOL=2,
	STRING=3
};

/**
 * Kinds of compositions
 */
enum CompKind {
	EACH_WITHIN=0,
	FIRST_WITHIN=1,
	LAST_WITHIN=2,
	ALL_WITHIN=3
};

/**
 * Operations used in constraints
 */
enum Op {
	EQ=0,
	LT=1,
	GT=2,
	NE=3,
	IN=4,
	LE=5,
	GE=6
};

#define OP_NUM 5

/**
 * Aggregate functions defined
 */
enum AggregateFun {
	NONE=0,
	AVG=1,
	COUNT=2,
	MIN=3,
	MAX=4,
	SUM=5
};

/**
 * Type of the state
 */
enum StateType {
	STATE=0,
	NEG=1,
	AGG=2
};

/**
 * Type of the reference
 * RULEPKT: refers to an aggregate or an attribute of an event composing the sequence
 * STATIC: describes a value, that can be an INT, FLOAT, BOOL, STRING
 */
enum ValRefType {
  RULEPKT = 0,
  STATIC = 1
};

enum LoopKind {
  SINGLE = 0,
  SINGLEGLOBAL = 1,
  SINGLENEG = 2,
  MULTIPLE = 3,
  MULTIPLENEG = 4
};

#endif
