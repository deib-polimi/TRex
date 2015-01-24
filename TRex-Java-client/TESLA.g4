grammar TESLA;

/*
@header {
    package polimi.trex.ruleparser;
}
*/

@members {
StringBuilder buf = new StringBuilder(); // can't make locals in lexer rules
}

ASSIGN     : 'Assign';
DEFINE     : 'Define';
FROM 	   : 'From';
WHERE 	   : 'Where';
CONSUMING  : 'Consuming';
VALTYPE    : 'string' | 'int' | 'float' | 'bool' ;
SEL_POLICY : 'each' | 'last' | 'first' ;
AGGR_FUN   : 'AVG' | 'SUM' | 'MAX' | 'MIN' | 'COUNT' ;
OPERATOR   : '=' | '>' | '<' | '>=' | '<=' | '!=' | '&' | '|' ;
BINOP_MUL  : '*' | '/';
BINOP_ADD  : '+' | '-' ;
INT_VAL    : ('0' .. '9')+;
FLOAT_VAL  : ('0' .. '9')+ '.' ('0' .. '9')+ ;
BOOL_VAL   : 'false' | 'true' ;
STRING_VAL :   '"'
        (   '\\'
            (   'r'     {buf.append('\r');}
            |   'n'     {buf.append('\n');}
            |   't'     {buf.append('\t');}
            |   '\\'    {buf.append('\\');}
            |   '\"'   {buf.append('"');}
            )
        |   ~('\\'|'"') {buf.append((char)_input.LA(-1));}
        )*
        '"'
        {setText(buf.toString()); buf.setLength(0); System.out.println(getText());}
    ;
EVT_NAME   : ('A' .. 'Z') (('A' .. 'Z') | ('a' .. 'z') | ('0' .. '9') | '_')*;
ATTR_NAME  : ('a' .. 'z') (('A' .. 'Z') | ('a' .. 'z') | ('0' .. '9') | '_')*;
PARAM_NAME : '$' ('a' .. 'z') (('A' .. 'Z') | ('a' .. 'z') | ('0' .. '9') | '_')*;
WS 	   : [ \t\r\n]+ -> skip ;

static_reference : (INT_VAL | FLOAT_VAL | STRING_VAL | BOOL_VAL);
packet_reference : (EVT_NAME '.' ATTR_NAME);
param_mapping: ATTR_NAME '=>' PARAM_NAME;
param_atom : (packet_reference | PARAM_NAME | static_reference); 
agg_one_reference : ('within' INT_VAL 'from' EVT_NAME);
agg_between : ('between' EVT_NAME 'and' EVT_NAME);
aggregate_atom : AGGR_FUN '(' packet_reference '(' ((attr_parameter | attr_constraint) (',' (attr_parameter | attr_constraint))* )? ')' ')' (agg_one_reference | agg_between) ; 
expr: expr BINOP_MUL expr | expr BINOP_ADD expr | '(' expr ')' | (param_atom | aggregate_atom);
attr_declaration : ATTR_NAME ':' VALTYPE;
staticAttr_definition: ATTR_NAME ':=' static_reference;
attr_definition: ATTR_NAME ':=' expr;
attr_constraint: ATTR_NAME OPERATOR static_reference;
attr_parameter: '[' VALTYPE ']' ATTR_NAME OPERATOR expr;
predicate : EVT_NAME '(' ((param_mapping | attr_constraint | attr_parameter) (',' (param_mapping | attr_constraint | attr_parameter))*)? ')' event_alias? ;
event_alias : 'as' EVT_NAME;
terminator : predicate;
positive_predicate : 'and' SEL_POLICY predicate 'within' INT_VAL 'from' EVT_NAME;
neg_one_reference: ('within' INT_VAL 'from' EVT_NAME);
neg_between: ('between' EVT_NAME 'and' EVT_NAME);
negative_predicate : 'and' 'not' predicate (neg_one_reference | neg_between);
pattern_predicate : positive_predicate | negative_predicate;
event_declaration : INT_VAL '=>' EVT_NAME;
event_declarations : event_declaration (',' event_declaration)*;
ce_definition : EVT_NAME '(' (attr_declaration (',' attr_declaration)* )? ')'; 
pattern : terminator (pattern_predicate)*;
definitions : (staticAttr_definition | attr_definition) (',' (staticAttr_definition | attr_definition))*;
consuming : EVT_NAME (',' EVT_NAME)*;
ending_rule : ';';
trex_rule : ASSIGN event_declarations DEFINE ce_definition FROM pattern (WHERE definitions)? (CONSUMING consuming)? ending_rule;