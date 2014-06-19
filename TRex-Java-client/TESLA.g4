grammar TESLA;

@header {
    package polimi.trex.client.ruleparser;
}


DEFINE : 'define';
FROM : 'from';
WHERE : 'where';
WITHIN : 'within';
CONSUMING : 'consuming';
BOOL_VAL: 'false' | 'true';
VALTYPE: ('string' | 'int' | 'float' | 'bool');
SEL_POLICY: ('each' | 'last' | 'first');
AGGR_FUN: 'AVG' | 'SUM' | 'MAX' | 'MIN' | 'COUNT';
OPERATOR: '=' | '>' | '<' | '!=' | '&' | '|';
BINOP_MUL: '*' | '/';
BINOP_ADD: '+' | '-' ;
FLOAT: ('0' .. '9')+ '.' ('0' .. '9')+;
INTEGER: ('0' .. '9')+;
STRING_VAL: '"' (('a' .. 'z') | ('A' .. 'Z') | ('0' .. '9'))* '"';
PRED_NAME : ('A' .. 'Z') ('a' .. 'z')*;
ATTR_NAME : ('a' .. 'z')+;
PARAM_NAME: '$' ('a' .. 'z')+;
WS : [ \t\r\n]+ -> skip ;

static_reference : (INTEGER | FLOAT | STRING_VAL | BOOL_VAL);
packet_reference : (PRED_NAME '.' ATTR_NAME);
param_mapping: ATTR_NAME '=>' PARAM_NAME;
param_atom : (packet_reference | PARAM_NAME | static_reference); 
agg_one_reference : (WITHIN INTEGER FROM PRED_NAME);
agg_between : ('between' PRED_NAME 'and' PRED_NAME);
aggregate_atom : AGGR_FUN '(' packet_reference '(' ((attr_parameter | attr_constraint) (',' (attr_parameter | attr_constraint))* )? ')' ')' (agg_one_reference | agg_between) ; 
expr: expr BINOP_MUL expr | expr BINOP_ADD expr | '(' expr ')' | (param_atom | aggregate_atom);
attr_declaration : ATTR_NAME ':' VALTYPE;
staticAttr_definition: ATTR_NAME ':=' static_reference;
attr_definition: ATTR_NAME ':=' expr;
attr_constraint: ATTR_NAME OPERATOR static_reference;
attr_parameter: '[' VALTYPE ']' ATTR_NAME OPERATOR expr;
ce_definition : PRED_NAME '(' (attr_declaration (',' attr_declaration)* )? ')'; 
predicate : PRED_NAME '(' ((param_mapping | attr_constraint | attr_parameter) (',' (param_mapping | attr_constraint | attr_parameter))*)? ')';
terminator : predicate;
positive_predicate : 'and' SEL_POLICY predicate WITHIN INTEGER FROM PRED_NAME;
neg_one_reference: (WITHIN INTEGER FROM PRED_NAME);
neg_between: ('between' PRED_NAME 'and' PRED_NAME);
negative_predicate : 'and not' predicate (neg_one_reference | neg_between);
pattern_predicate : positive_predicate | negative_predicate;
definitions : (staticAttr_definition | attr_definition) (',' (staticAttr_definition | attr_definition))*;
consuming : PRED_NAME (',' PRED_NAME)*;
pattern: terminator (pattern_predicate)*;
ending_rule: ';';
trex_rule : DEFINE ce_definition FROM pattern (WHERE definitions)? (CONSUMING consuming)? ending_rule;