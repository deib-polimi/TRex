// Generated from TESLA.g4 by ANTLR 4.5
package polimi.trex.ruleparser;
import org.antlr.v4.runtime.misc.NotNull;
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link TESLAParser}.
 */
public interface TESLAListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link TESLAParser#static_reference}.
	 * @param ctx the parse tree
	 */
	void enterStatic_reference(TESLAParser.Static_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#static_reference}.
	 * @param ctx the parse tree
	 */
	void exitStatic_reference(TESLAParser.Static_referenceContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#packet_reference}.
	 * @param ctx the parse tree
	 */
	void enterPacket_reference(TESLAParser.Packet_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#packet_reference}.
	 * @param ctx the parse tree
	 */
	void exitPacket_reference(TESLAParser.Packet_referenceContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#param_mapping}.
	 * @param ctx the parse tree
	 */
	void enterParam_mapping(TESLAParser.Param_mappingContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#param_mapping}.
	 * @param ctx the parse tree
	 */
	void exitParam_mapping(TESLAParser.Param_mappingContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#param_atom}.
	 * @param ctx the parse tree
	 */
	void enterParam_atom(TESLAParser.Param_atomContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#param_atom}.
	 * @param ctx the parse tree
	 */
	void exitParam_atom(TESLAParser.Param_atomContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#agg_one_reference}.
	 * @param ctx the parse tree
	 */
	void enterAgg_one_reference(TESLAParser.Agg_one_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#agg_one_reference}.
	 * @param ctx the parse tree
	 */
	void exitAgg_one_reference(TESLAParser.Agg_one_referenceContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#agg_between}.
	 * @param ctx the parse tree
	 */
	void enterAgg_between(TESLAParser.Agg_betweenContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#agg_between}.
	 * @param ctx the parse tree
	 */
	void exitAgg_between(TESLAParser.Agg_betweenContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#aggregate_atom}.
	 * @param ctx the parse tree
	 */
	void enterAggregate_atom(TESLAParser.Aggregate_atomContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#aggregate_atom}.
	 * @param ctx the parse tree
	 */
	void exitAggregate_atom(TESLAParser.Aggregate_atomContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterExpr(TESLAParser.ExprContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitExpr(TESLAParser.ExprContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_declaration}.
	 * @param ctx the parse tree
	 */
	void enterAttr_declaration(TESLAParser.Attr_declarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_declaration}.
	 * @param ctx the parse tree
	 */
	void exitAttr_declaration(TESLAParser.Attr_declarationContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#staticAttr_definition}.
	 * @param ctx the parse tree
	 */
	void enterStaticAttr_definition(TESLAParser.StaticAttr_definitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#staticAttr_definition}.
	 * @param ctx the parse tree
	 */
	void exitStaticAttr_definition(TESLAParser.StaticAttr_definitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_definition}.
	 * @param ctx the parse tree
	 */
	void enterAttr_definition(TESLAParser.Attr_definitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_definition}.
	 * @param ctx the parse tree
	 */
	void exitAttr_definition(TESLAParser.Attr_definitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_constraint}.
	 * @param ctx the parse tree
	 */
	void enterAttr_constraint(TESLAParser.Attr_constraintContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_constraint}.
	 * @param ctx the parse tree
	 */
	void exitAttr_constraint(TESLAParser.Attr_constraintContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_parameter}.
	 * @param ctx the parse tree
	 */
	void enterAttr_parameter(TESLAParser.Attr_parameterContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_parameter}.
	 * @param ctx the parse tree
	 */
	void exitAttr_parameter(TESLAParser.Attr_parameterContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#predicate}.
	 * @param ctx the parse tree
	 */
	void enterPredicate(TESLAParser.PredicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#predicate}.
	 * @param ctx the parse tree
	 */
	void exitPredicate(TESLAParser.PredicateContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#event_alias}.
	 * @param ctx the parse tree
	 */
	void enterEvent_alias(TESLAParser.Event_aliasContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#event_alias}.
	 * @param ctx the parse tree
	 */
	void exitEvent_alias(TESLAParser.Event_aliasContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#terminator}.
	 * @param ctx the parse tree
	 */
	void enterTerminator(TESLAParser.TerminatorContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#terminator}.
	 * @param ctx the parse tree
	 */
	void exitTerminator(TESLAParser.TerminatorContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#positive_predicate}.
	 * @param ctx the parse tree
	 */
	void enterPositive_predicate(TESLAParser.Positive_predicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#positive_predicate}.
	 * @param ctx the parse tree
	 */
	void exitPositive_predicate(TESLAParser.Positive_predicateContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#neg_one_reference}.
	 * @param ctx the parse tree
	 */
	void enterNeg_one_reference(TESLAParser.Neg_one_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#neg_one_reference}.
	 * @param ctx the parse tree
	 */
	void exitNeg_one_reference(TESLAParser.Neg_one_referenceContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#neg_between}.
	 * @param ctx the parse tree
	 */
	void enterNeg_between(TESLAParser.Neg_betweenContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#neg_between}.
	 * @param ctx the parse tree
	 */
	void exitNeg_between(TESLAParser.Neg_betweenContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#negative_predicate}.
	 * @param ctx the parse tree
	 */
	void enterNegative_predicate(TESLAParser.Negative_predicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#negative_predicate}.
	 * @param ctx the parse tree
	 */
	void exitNegative_predicate(TESLAParser.Negative_predicateContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#pattern_predicate}.
	 * @param ctx the parse tree
	 */
	void enterPattern_predicate(TESLAParser.Pattern_predicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#pattern_predicate}.
	 * @param ctx the parse tree
	 */
	void exitPattern_predicate(TESLAParser.Pattern_predicateContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#event_declaration}.
	 * @param ctx the parse tree
	 */
	void enterEvent_declaration(TESLAParser.Event_declarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#event_declaration}.
	 * @param ctx the parse tree
	 */
	void exitEvent_declaration(TESLAParser.Event_declarationContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#event_declarations}.
	 * @param ctx the parse tree
	 */
	void enterEvent_declarations(TESLAParser.Event_declarationsContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#event_declarations}.
	 * @param ctx the parse tree
	 */
	void exitEvent_declarations(TESLAParser.Event_declarationsContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#ce_definition}.
	 * @param ctx the parse tree
	 */
	void enterCe_definition(TESLAParser.Ce_definitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#ce_definition}.
	 * @param ctx the parse tree
	 */
	void exitCe_definition(TESLAParser.Ce_definitionContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#pattern}.
	 * @param ctx the parse tree
	 */
	void enterPattern(TESLAParser.PatternContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#pattern}.
	 * @param ctx the parse tree
	 */
	void exitPattern(TESLAParser.PatternContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#definitions}.
	 * @param ctx the parse tree
	 */
	void enterDefinitions(TESLAParser.DefinitionsContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#definitions}.
	 * @param ctx the parse tree
	 */
	void exitDefinitions(TESLAParser.DefinitionsContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#consuming}.
	 * @param ctx the parse tree
	 */
	void enterConsuming(TESLAParser.ConsumingContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#consuming}.
	 * @param ctx the parse tree
	 */
	void exitConsuming(TESLAParser.ConsumingContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#ending_rule}.
	 * @param ctx the parse tree
	 */
	void enterEnding_rule(TESLAParser.Ending_ruleContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#ending_rule}.
	 * @param ctx the parse tree
	 */
	void exitEnding_rule(TESLAParser.Ending_ruleContext ctx);
	/**
	 * Enter a parse tree produced by {@link TESLAParser#trex_rule}.
	 * @param ctx the parse tree
	 */
	void enterTrex_rule(TESLAParser.Trex_ruleContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#trex_rule}.
	 * @param ctx the parse tree
	 */
	void exitTrex_rule(TESLAParser.Trex_ruleContext ctx);
}