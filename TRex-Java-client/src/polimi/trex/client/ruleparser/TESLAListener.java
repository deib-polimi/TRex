// Generated from TESLA.g4 by ANTLR 4.2.2

    package polimi.trex.client.ruleparser;

import org.antlr.v4.runtime.misc.NotNull;
import org.antlr.v4.runtime.tree.ParseTreeListener;

/**
 * This interface defines a complete listener for a parse tree produced by
 * {@link TESLAParser}.
 */
public interface TESLAListener extends ParseTreeListener {
	/**
	 * Enter a parse tree produced by {@link TESLAParser#trex_rule}.
	 * @param ctx the parse tree
	 */
	void enterTrex_rule(@NotNull TESLAParser.Trex_ruleContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#trex_rule}.
	 * @param ctx the parse tree
	 */
	void exitTrex_rule(@NotNull TESLAParser.Trex_ruleContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#neg_between}.
	 * @param ctx the parse tree
	 */
	void enterNeg_between(@NotNull TESLAParser.Neg_betweenContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#neg_between}.
	 * @param ctx the parse tree
	 */
	void exitNeg_between(@NotNull TESLAParser.Neg_betweenContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#agg_between}.
	 * @param ctx the parse tree
	 */
	void enterAgg_between(@NotNull TESLAParser.Agg_betweenContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#agg_between}.
	 * @param ctx the parse tree
	 */
	void exitAgg_between(@NotNull TESLAParser.Agg_betweenContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#static_reference}.
	 * @param ctx the parse tree
	 */
	void enterStatic_reference(@NotNull TESLAParser.Static_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#static_reference}.
	 * @param ctx the parse tree
	 */
	void exitStatic_reference(@NotNull TESLAParser.Static_referenceContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#negative_predicate}.
	 * @param ctx the parse tree
	 */
	void enterNegative_predicate(@NotNull TESLAParser.Negative_predicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#negative_predicate}.
	 * @param ctx the parse tree
	 */
	void exitNegative_predicate(@NotNull TESLAParser.Negative_predicateContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#ce_definition}.
	 * @param ctx the parse tree
	 */
	void enterCe_definition(@NotNull TESLAParser.Ce_definitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#ce_definition}.
	 * @param ctx the parse tree
	 */
	void exitCe_definition(@NotNull TESLAParser.Ce_definitionContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#predicate}.
	 * @param ctx the parse tree
	 */
	void enterPredicate(@NotNull TESLAParser.PredicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#predicate}.
	 * @param ctx the parse tree
	 */
	void exitPredicate(@NotNull TESLAParser.PredicateContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#expr}.
	 * @param ctx the parse tree
	 */
	void enterExpr(@NotNull TESLAParser.ExprContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#expr}.
	 * @param ctx the parse tree
	 */
	void exitExpr(@NotNull TESLAParser.ExprContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_parameter}.
	 * @param ctx the parse tree
	 */
	void enterAttr_parameter(@NotNull TESLAParser.Attr_parameterContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_parameter}.
	 * @param ctx the parse tree
	 */
	void exitAttr_parameter(@NotNull TESLAParser.Attr_parameterContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_constraint}.
	 * @param ctx the parse tree
	 */
	void enterAttr_constraint(@NotNull TESLAParser.Attr_constraintContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_constraint}.
	 * @param ctx the parse tree
	 */
	void exitAttr_constraint(@NotNull TESLAParser.Attr_constraintContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#consuming}.
	 * @param ctx the parse tree
	 */
	void enterConsuming(@NotNull TESLAParser.ConsumingContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#consuming}.
	 * @param ctx the parse tree
	 */
	void exitConsuming(@NotNull TESLAParser.ConsumingContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_declaration}.
	 * @param ctx the parse tree
	 */
	void enterAttr_declaration(@NotNull TESLAParser.Attr_declarationContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_declaration}.
	 * @param ctx the parse tree
	 */
	void exitAttr_declaration(@NotNull TESLAParser.Attr_declarationContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#aggregate_atom}.
	 * @param ctx the parse tree
	 */
	void enterAggregate_atom(@NotNull TESLAParser.Aggregate_atomContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#aggregate_atom}.
	 * @param ctx the parse tree
	 */
	void exitAggregate_atom(@NotNull TESLAParser.Aggregate_atomContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#pattern_predicate}.
	 * @param ctx the parse tree
	 */
	void enterPattern_predicate(@NotNull TESLAParser.Pattern_predicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#pattern_predicate}.
	 * @param ctx the parse tree
	 */
	void exitPattern_predicate(@NotNull TESLAParser.Pattern_predicateContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#agg_one_reference}.
	 * @param ctx the parse tree
	 */
	void enterAgg_one_reference(@NotNull TESLAParser.Agg_one_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#agg_one_reference}.
	 * @param ctx the parse tree
	 */
	void exitAgg_one_reference(@NotNull TESLAParser.Agg_one_referenceContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#positive_predicate}.
	 * @param ctx the parse tree
	 */
	void enterPositive_predicate(@NotNull TESLAParser.Positive_predicateContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#positive_predicate}.
	 * @param ctx the parse tree
	 */
	void exitPositive_predicate(@NotNull TESLAParser.Positive_predicateContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#terminator}.
	 * @param ctx the parse tree
	 */
	void enterTerminator(@NotNull TESLAParser.TerminatorContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#terminator}.
	 * @param ctx the parse tree
	 */
	void exitTerminator(@NotNull TESLAParser.TerminatorContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#param_mapping}.
	 * @param ctx the parse tree
	 */
	void enterParam_mapping(@NotNull TESLAParser.Param_mappingContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#param_mapping}.
	 * @param ctx the parse tree
	 */
	void exitParam_mapping(@NotNull TESLAParser.Param_mappingContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#definitions}.
	 * @param ctx the parse tree
	 */
	void enterDefinitions(@NotNull TESLAParser.DefinitionsContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#definitions}.
	 * @param ctx the parse tree
	 */
	void exitDefinitions(@NotNull TESLAParser.DefinitionsContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#packet_reference}.
	 * @param ctx the parse tree
	 */
	void enterPacket_reference(@NotNull TESLAParser.Packet_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#packet_reference}.
	 * @param ctx the parse tree
	 */
	void exitPacket_reference(@NotNull TESLAParser.Packet_referenceContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#attr_definition}.
	 * @param ctx the parse tree
	 */
	void enterAttr_definition(@NotNull TESLAParser.Attr_definitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#attr_definition}.
	 * @param ctx the parse tree
	 */
	void exitAttr_definition(@NotNull TESLAParser.Attr_definitionContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#param_atom}.
	 * @param ctx the parse tree
	 */
	void enterParam_atom(@NotNull TESLAParser.Param_atomContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#param_atom}.
	 * @param ctx the parse tree
	 */
	void exitParam_atom(@NotNull TESLAParser.Param_atomContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#neg_one_reference}.
	 * @param ctx the parse tree
	 */
	void enterNeg_one_reference(@NotNull TESLAParser.Neg_one_referenceContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#neg_one_reference}.
	 * @param ctx the parse tree
	 */
	void exitNeg_one_reference(@NotNull TESLAParser.Neg_one_referenceContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#pattern}.
	 * @param ctx the parse tree
	 */
	void enterPattern(@NotNull TESLAParser.PatternContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#pattern}.
	 * @param ctx the parse tree
	 */
	void exitPattern(@NotNull TESLAParser.PatternContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#ending_rule}.
	 * @param ctx the parse tree
	 */
	void enterEnding_rule(@NotNull TESLAParser.Ending_ruleContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#ending_rule}.
	 * @param ctx the parse tree
	 */
	void exitEnding_rule(@NotNull TESLAParser.Ending_ruleContext ctx);

	/**
	 * Enter a parse tree produced by {@link TESLAParser#staticAttr_definition}.
	 * @param ctx the parse tree
	 */
	void enterStaticAttr_definition(@NotNull TESLAParser.StaticAttr_definitionContext ctx);
	/**
	 * Exit a parse tree produced by {@link TESLAParser#staticAttr_definition}.
	 * @param ctx the parse tree
	 */
	void exitStaticAttr_definition(@NotNull TESLAParser.StaticAttr_definitionContext ctx);
}