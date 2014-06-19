// Generated from TESLA.g4 by ANTLR 4.2.2

    package polimi.trex.client.ruleparser;

import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class TESLAParser extends Parser {
	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		T__12=1, T__11=2, T__10=3, T__9=4, T__8=5, T__7=6, T__6=7, T__5=8, T__4=9, 
		T__3=10, T__2=11, T__1=12, T__0=13, DEFINE=14, FROM=15, WHERE=16, WITHIN=17, 
		CONSUMING=18, BOOL_VAL=19, VALTYPE=20, SEL_POLICY=21, AGGR_FUN=22, OPERATOR=23, 
		BINOP_MUL=24, BINOP_ADD=25, FLOAT=26, INTEGER=27, STRING_VAL=28, PRED_NAME=29, 
		ATTR_NAME=30, PARAM_NAME=31, WS=32;
	public static final String[] tokenNames = {
		"<INVALID>", "']'", "')'", "'=>'", "'.'", "','", "'['", "':'", "'('", 
		"';'", "'and'", "':='", "'and not'", "'between'", "'define'", "'from'", 
		"'where'", "'within'", "'consuming'", "BOOL_VAL", "VALTYPE", "SEL_POLICY", 
		"AGGR_FUN", "OPERATOR", "BINOP_MUL", "BINOP_ADD", "FLOAT", "INTEGER", 
		"STRING_VAL", "PRED_NAME", "ATTR_NAME", "PARAM_NAME", "WS"
	};
	public static final int
		RULE_static_reference = 0, RULE_packet_reference = 1, RULE_param_mapping = 2, 
		RULE_param_atom = 3, RULE_agg_one_reference = 4, RULE_agg_between = 5, 
		RULE_aggregate_atom = 6, RULE_expr = 7, RULE_attr_declaration = 8, RULE_staticAttr_definition = 9, 
		RULE_attr_definition = 10, RULE_attr_constraint = 11, RULE_attr_parameter = 12, 
		RULE_ce_definition = 13, RULE_predicate = 14, RULE_terminator = 15, RULE_positive_predicate = 16, 
		RULE_neg_one_reference = 17, RULE_neg_between = 18, RULE_negative_predicate = 19, 
		RULE_pattern_predicate = 20, RULE_definitions = 21, RULE_consuming = 22, 
		RULE_pattern = 23, RULE_ending_rule = 24, RULE_trex_rule = 25;
	public static final String[] ruleNames = {
		"static_reference", "packet_reference", "param_mapping", "param_atom", 
		"agg_one_reference", "agg_between", "aggregate_atom", "expr", "attr_declaration", 
		"staticAttr_definition", "attr_definition", "attr_constraint", "attr_parameter", 
		"ce_definition", "predicate", "terminator", "positive_predicate", "neg_one_reference", 
		"neg_between", "negative_predicate", "pattern_predicate", "definitions", 
		"consuming", "pattern", "ending_rule", "trex_rule"
	};

	@Override
	public String getGrammarFileName() { return "TESLA.g4"; }

	@Override
	public String[] getTokenNames() { return tokenNames; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public TESLAParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}
	public static class Static_referenceContext extends ParserRuleContext {
		public TerminalNode FLOAT() { return getToken(TESLAParser.FLOAT, 0); }
		public TerminalNode STRING_VAL() { return getToken(TESLAParser.STRING_VAL, 0); }
		public TerminalNode BOOL_VAL() { return getToken(TESLAParser.BOOL_VAL, 0); }
		public TerminalNode INTEGER() { return getToken(TESLAParser.INTEGER, 0); }
		public Static_referenceContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_static_reference; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterStatic_reference(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitStatic_reference(this);
		}
	}

	public final Static_referenceContext static_reference() throws RecognitionException {
		Static_referenceContext _localctx = new Static_referenceContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_static_reference);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(52);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << BOOL_VAL) | (1L << FLOAT) | (1L << INTEGER) | (1L << STRING_VAL))) != 0)) ) {
			_errHandler.recoverInline(this);
			}
			consume();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Packet_referenceContext extends ParserRuleContext {
		public TerminalNode PRED_NAME() { return getToken(TESLAParser.PRED_NAME, 0); }
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public Packet_referenceContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_packet_reference; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterPacket_reference(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitPacket_reference(this);
		}
	}

	public final Packet_referenceContext packet_reference() throws RecognitionException {
		Packet_referenceContext _localctx = new Packet_referenceContext(_ctx, getState());
		enterRule(_localctx, 2, RULE_packet_reference);
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(54); match(PRED_NAME);
			setState(55); match(4);
			setState(56); match(ATTR_NAME);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Param_mappingContext extends ParserRuleContext {
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public TerminalNode PARAM_NAME() { return getToken(TESLAParser.PARAM_NAME, 0); }
		public Param_mappingContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_param_mapping; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterParam_mapping(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitParam_mapping(this);
		}
	}

	public final Param_mappingContext param_mapping() throws RecognitionException {
		Param_mappingContext _localctx = new Param_mappingContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_param_mapping);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(58); match(ATTR_NAME);
			setState(59); match(3);
			setState(60); match(PARAM_NAME);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Param_atomContext extends ParserRuleContext {
		public Static_referenceContext static_reference() {
			return getRuleContext(Static_referenceContext.class,0);
		}
		public Packet_referenceContext packet_reference() {
			return getRuleContext(Packet_referenceContext.class,0);
		}
		public TerminalNode PARAM_NAME() { return getToken(TESLAParser.PARAM_NAME, 0); }
		public Param_atomContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_param_atom; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterParam_atom(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitParam_atom(this);
		}
	}

	public final Param_atomContext param_atom() throws RecognitionException {
		Param_atomContext _localctx = new Param_atomContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_param_atom);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(65);
			switch (_input.LA(1)) {
			case PRED_NAME:
				{
				setState(62); packet_reference();
				}
				break;
			case PARAM_NAME:
				{
				setState(63); match(PARAM_NAME);
				}
				break;
			case BOOL_VAL:
			case FLOAT:
			case INTEGER:
			case STRING_VAL:
				{
				setState(64); static_reference();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Agg_one_referenceContext extends ParserRuleContext {
		public TerminalNode PRED_NAME() { return getToken(TESLAParser.PRED_NAME, 0); }
		public TerminalNode WITHIN() { return getToken(TESLAParser.WITHIN, 0); }
		public TerminalNode INTEGER() { return getToken(TESLAParser.INTEGER, 0); }
		public TerminalNode FROM() { return getToken(TESLAParser.FROM, 0); }
		public Agg_one_referenceContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_agg_one_reference; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAgg_one_reference(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAgg_one_reference(this);
		}
	}

	public final Agg_one_referenceContext agg_one_reference() throws RecognitionException {
		Agg_one_referenceContext _localctx = new Agg_one_referenceContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_agg_one_reference);
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(67); match(WITHIN);
			setState(68); match(INTEGER);
			setState(69); match(FROM);
			setState(70); match(PRED_NAME);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Agg_betweenContext extends ParserRuleContext {
		public List<TerminalNode> PRED_NAME() { return getTokens(TESLAParser.PRED_NAME); }
		public TerminalNode PRED_NAME(int i) {
			return getToken(TESLAParser.PRED_NAME, i);
		}
		public Agg_betweenContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_agg_between; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAgg_between(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAgg_between(this);
		}
	}

	public final Agg_betweenContext agg_between() throws RecognitionException {
		Agg_betweenContext _localctx = new Agg_betweenContext(_ctx, getState());
		enterRule(_localctx, 10, RULE_agg_between);
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(72); match(13);
			setState(73); match(PRED_NAME);
			setState(74); match(10);
			setState(75); match(PRED_NAME);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Aggregate_atomContext extends ParserRuleContext {
		public List<Attr_parameterContext> attr_parameter() {
			return getRuleContexts(Attr_parameterContext.class);
		}
		public Agg_betweenContext agg_between() {
			return getRuleContext(Agg_betweenContext.class,0);
		}
		public Agg_one_referenceContext agg_one_reference() {
			return getRuleContext(Agg_one_referenceContext.class,0);
		}
		public Attr_parameterContext attr_parameter(int i) {
			return getRuleContext(Attr_parameterContext.class,i);
		}
		public Attr_constraintContext attr_constraint(int i) {
			return getRuleContext(Attr_constraintContext.class,i);
		}
		public Packet_referenceContext packet_reference() {
			return getRuleContext(Packet_referenceContext.class,0);
		}
		public TerminalNode AGGR_FUN() { return getToken(TESLAParser.AGGR_FUN, 0); }
		public List<Attr_constraintContext> attr_constraint() {
			return getRuleContexts(Attr_constraintContext.class);
		}
		public Aggregate_atomContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_aggregate_atom; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAggregate_atom(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAggregate_atom(this);
		}
	}

	public final Aggregate_atomContext aggregate_atom() throws RecognitionException {
		Aggregate_atomContext _localctx = new Aggregate_atomContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_aggregate_atom);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(77); match(AGGR_FUN);
			setState(78); match(8);
			setState(79); packet_reference();
			setState(80); match(8);
			setState(95);
			_la = _input.LA(1);
			if (_la==6 || _la==ATTR_NAME) {
				{
				setState(83);
				switch (_input.LA(1)) {
				case 6:
					{
					setState(81); attr_parameter();
					}
					break;
				case ATTR_NAME:
					{
					setState(82); attr_constraint();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(92);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==5) {
					{
					{
					setState(85); match(5);
					setState(88);
					switch (_input.LA(1)) {
					case 6:
						{
						setState(86); attr_parameter();
						}
						break;
					case ATTR_NAME:
						{
						setState(87); attr_constraint();
						}
						break;
					default:
						throw new NoViableAltException(this);
					}
					}
					}
					setState(94);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(97); match(2);
			setState(98); match(2);
			setState(101);
			switch (_input.LA(1)) {
			case WITHIN:
				{
				setState(99); agg_one_reference();
				}
				break;
			case 13:
				{
				setState(100); agg_between();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ExprContext extends ParserRuleContext {
		public TerminalNode BINOP_MUL() { return getToken(TESLAParser.BINOP_MUL, 0); }
		public Param_atomContext param_atom() {
			return getRuleContext(Param_atomContext.class,0);
		}
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public Aggregate_atomContext aggregate_atom() {
			return getRuleContext(Aggregate_atomContext.class,0);
		}
		public TerminalNode BINOP_ADD() { return getToken(TESLAParser.BINOP_ADD, 0); }
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterExpr(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitExpr(this);
		}
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 14;
		enterRecursionRule(_localctx, 14, RULE_expr, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(112);
			switch (_input.LA(1)) {
			case 8:
				{
				setState(104); match(8);
				setState(105); expr(0);
				setState(106); match(2);
				}
				break;
			case BOOL_VAL:
			case AGGR_FUN:
			case FLOAT:
			case INTEGER:
			case STRING_VAL:
			case PRED_NAME:
			case PARAM_NAME:
				{
				setState(110);
				switch (_input.LA(1)) {
				case BOOL_VAL:
				case FLOAT:
				case INTEGER:
				case STRING_VAL:
				case PRED_NAME:
				case PARAM_NAME:
					{
					setState(108); param_atom();
					}
					break;
				case AGGR_FUN:
					{
					setState(109); aggregate_atom();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			_ctx.stop = _input.LT(-1);
			setState(122);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			while ( _alt!=2 && _alt!=ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(120);
					switch ( getInterpreter().adaptivePredict(_input,8,_ctx) ) {
					case 1:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(114);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(115); match(BINOP_MUL);
						setState(116); expr(5);
						}
						break;

					case 2:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(117);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(118); match(BINOP_ADD);
						setState(119); expr(4);
						}
						break;
					}
					} 
				}
				setState(124);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class Attr_declarationContext extends ParserRuleContext {
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public TerminalNode VALTYPE() { return getToken(TESLAParser.VALTYPE, 0); }
		public Attr_declarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attr_declaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAttr_declaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAttr_declaration(this);
		}
	}

	public final Attr_declarationContext attr_declaration() throws RecognitionException {
		Attr_declarationContext _localctx = new Attr_declarationContext(_ctx, getState());
		enterRule(_localctx, 16, RULE_attr_declaration);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(125); match(ATTR_NAME);
			setState(126); match(7);
			setState(127); match(VALTYPE);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class StaticAttr_definitionContext extends ParserRuleContext {
		public Static_referenceContext static_reference() {
			return getRuleContext(Static_referenceContext.class,0);
		}
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public StaticAttr_definitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_staticAttr_definition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterStaticAttr_definition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitStaticAttr_definition(this);
		}
	}

	public final StaticAttr_definitionContext staticAttr_definition() throws RecognitionException {
		StaticAttr_definitionContext _localctx = new StaticAttr_definitionContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_staticAttr_definition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(129); match(ATTR_NAME);
			setState(130); match(11);
			setState(131); static_reference();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Attr_definitionContext extends ParserRuleContext {
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public Attr_definitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attr_definition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAttr_definition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAttr_definition(this);
		}
	}

	public final Attr_definitionContext attr_definition() throws RecognitionException {
		Attr_definitionContext _localctx = new Attr_definitionContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_attr_definition);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(133); match(ATTR_NAME);
			setState(134); match(11);
			setState(135); expr(0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Attr_constraintContext extends ParserRuleContext {
		public Static_referenceContext static_reference() {
			return getRuleContext(Static_referenceContext.class,0);
		}
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public TerminalNode OPERATOR() { return getToken(TESLAParser.OPERATOR, 0); }
		public Attr_constraintContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attr_constraint; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAttr_constraint(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAttr_constraint(this);
		}
	}

	public final Attr_constraintContext attr_constraint() throws RecognitionException {
		Attr_constraintContext _localctx = new Attr_constraintContext(_ctx, getState());
		enterRule(_localctx, 22, RULE_attr_constraint);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(137); match(ATTR_NAME);
			setState(138); match(OPERATOR);
			setState(139); static_reference();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Attr_parameterContext extends ParserRuleContext {
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode VALTYPE() { return getToken(TESLAParser.VALTYPE, 0); }
		public TerminalNode OPERATOR() { return getToken(TESLAParser.OPERATOR, 0); }
		public Attr_parameterContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_attr_parameter; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterAttr_parameter(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitAttr_parameter(this);
		}
	}

	public final Attr_parameterContext attr_parameter() throws RecognitionException {
		Attr_parameterContext _localctx = new Attr_parameterContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_attr_parameter);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(141); match(6);
			setState(142); match(VALTYPE);
			setState(143); match(1);
			setState(144); match(ATTR_NAME);
			setState(145); match(OPERATOR);
			setState(146); expr(0);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Ce_definitionContext extends ParserRuleContext {
		public TerminalNode PRED_NAME() { return getToken(TESLAParser.PRED_NAME, 0); }
		public Attr_declarationContext attr_declaration(int i) {
			return getRuleContext(Attr_declarationContext.class,i);
		}
		public List<Attr_declarationContext> attr_declaration() {
			return getRuleContexts(Attr_declarationContext.class);
		}
		public Ce_definitionContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ce_definition; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterCe_definition(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitCe_definition(this);
		}
	}

	public final Ce_definitionContext ce_definition() throws RecognitionException {
		Ce_definitionContext _localctx = new Ce_definitionContext(_ctx, getState());
		enterRule(_localctx, 26, RULE_ce_definition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(148); match(PRED_NAME);
			setState(149); match(8);
			setState(158);
			_la = _input.LA(1);
			if (_la==ATTR_NAME) {
				{
				setState(150); attr_declaration();
				setState(155);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==5) {
					{
					{
					setState(151); match(5);
					setState(152); attr_declaration();
					}
					}
					setState(157);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(160); match(2);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PredicateContext extends ParserRuleContext {
		public List<Attr_parameterContext> attr_parameter() {
			return getRuleContexts(Attr_parameterContext.class);
		}
		public Attr_parameterContext attr_parameter(int i) {
			return getRuleContext(Attr_parameterContext.class,i);
		}
		public TerminalNode PRED_NAME() { return getToken(TESLAParser.PRED_NAME, 0); }
		public Attr_constraintContext attr_constraint(int i) {
			return getRuleContext(Attr_constraintContext.class,i);
		}
		public Param_mappingContext param_mapping(int i) {
			return getRuleContext(Param_mappingContext.class,i);
		}
		public List<Param_mappingContext> param_mapping() {
			return getRuleContexts(Param_mappingContext.class);
		}
		public List<Attr_constraintContext> attr_constraint() {
			return getRuleContexts(Attr_constraintContext.class);
		}
		public PredicateContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_predicate; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterPredicate(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitPredicate(this);
		}
	}

	public final PredicateContext predicate() throws RecognitionException {
		PredicateContext _localctx = new PredicateContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_predicate);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(162); match(PRED_NAME);
			setState(163); match(8);
			setState(180);
			_la = _input.LA(1);
			if (_la==6 || _la==ATTR_NAME) {
				{
				setState(167);
				switch ( getInterpreter().adaptivePredict(_input,12,_ctx) ) {
				case 1:
					{
					setState(164); param_mapping();
					}
					break;

				case 2:
					{
					setState(165); attr_constraint();
					}
					break;

				case 3:
					{
					setState(166); attr_parameter();
					}
					break;
				}
				setState(177);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==5) {
					{
					{
					setState(169); match(5);
					setState(173);
					switch ( getInterpreter().adaptivePredict(_input,13,_ctx) ) {
					case 1:
						{
						setState(170); param_mapping();
						}
						break;

					case 2:
						{
						setState(171); attr_constraint();
						}
						break;

					case 3:
						{
						setState(172); attr_parameter();
						}
						break;
					}
					}
					}
					setState(179);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(182); match(2);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class TerminatorContext extends ParserRuleContext {
		public PredicateContext predicate() {
			return getRuleContext(PredicateContext.class,0);
		}
		public TerminatorContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_terminator; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterTerminator(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitTerminator(this);
		}
	}

	public final TerminatorContext terminator() throws RecognitionException {
		TerminatorContext _localctx = new TerminatorContext(_ctx, getState());
		enterRule(_localctx, 30, RULE_terminator);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(184); predicate();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Positive_predicateContext extends ParserRuleContext {
		public TerminalNode PRED_NAME() { return getToken(TESLAParser.PRED_NAME, 0); }
		public PredicateContext predicate() {
			return getRuleContext(PredicateContext.class,0);
		}
		public TerminalNode WITHIN() { return getToken(TESLAParser.WITHIN, 0); }
		public TerminalNode INTEGER() { return getToken(TESLAParser.INTEGER, 0); }
		public TerminalNode SEL_POLICY() { return getToken(TESLAParser.SEL_POLICY, 0); }
		public TerminalNode FROM() { return getToken(TESLAParser.FROM, 0); }
		public Positive_predicateContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_positive_predicate; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterPositive_predicate(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitPositive_predicate(this);
		}
	}

	public final Positive_predicateContext positive_predicate() throws RecognitionException {
		Positive_predicateContext _localctx = new Positive_predicateContext(_ctx, getState());
		enterRule(_localctx, 32, RULE_positive_predicate);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(186); match(10);
			setState(187); match(SEL_POLICY);
			setState(188); predicate();
			setState(189); match(WITHIN);
			setState(190); match(INTEGER);
			setState(191); match(FROM);
			setState(192); match(PRED_NAME);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Neg_one_referenceContext extends ParserRuleContext {
		public TerminalNode PRED_NAME() { return getToken(TESLAParser.PRED_NAME, 0); }
		public TerminalNode WITHIN() { return getToken(TESLAParser.WITHIN, 0); }
		public TerminalNode INTEGER() { return getToken(TESLAParser.INTEGER, 0); }
		public TerminalNode FROM() { return getToken(TESLAParser.FROM, 0); }
		public Neg_one_referenceContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_neg_one_reference; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterNeg_one_reference(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitNeg_one_reference(this);
		}
	}

	public final Neg_one_referenceContext neg_one_reference() throws RecognitionException {
		Neg_one_referenceContext _localctx = new Neg_one_referenceContext(_ctx, getState());
		enterRule(_localctx, 34, RULE_neg_one_reference);
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(194); match(WITHIN);
			setState(195); match(INTEGER);
			setState(196); match(FROM);
			setState(197); match(PRED_NAME);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Neg_betweenContext extends ParserRuleContext {
		public List<TerminalNode> PRED_NAME() { return getTokens(TESLAParser.PRED_NAME); }
		public TerminalNode PRED_NAME(int i) {
			return getToken(TESLAParser.PRED_NAME, i);
		}
		public Neg_betweenContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_neg_between; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterNeg_between(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitNeg_between(this);
		}
	}

	public final Neg_betweenContext neg_between() throws RecognitionException {
		Neg_betweenContext _localctx = new Neg_betweenContext(_ctx, getState());
		enterRule(_localctx, 36, RULE_neg_between);
		try {
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(199); match(13);
			setState(200); match(PRED_NAME);
			setState(201); match(10);
			setState(202); match(PRED_NAME);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Negative_predicateContext extends ParserRuleContext {
		public Neg_betweenContext neg_between() {
			return getRuleContext(Neg_betweenContext.class,0);
		}
		public PredicateContext predicate() {
			return getRuleContext(PredicateContext.class,0);
		}
		public Neg_one_referenceContext neg_one_reference() {
			return getRuleContext(Neg_one_referenceContext.class,0);
		}
		public Negative_predicateContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_negative_predicate; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterNegative_predicate(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitNegative_predicate(this);
		}
	}

	public final Negative_predicateContext negative_predicate() throws RecognitionException {
		Negative_predicateContext _localctx = new Negative_predicateContext(_ctx, getState());
		enterRule(_localctx, 38, RULE_negative_predicate);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(204); match(12);
			setState(205); predicate();
			setState(208);
			switch (_input.LA(1)) {
			case WITHIN:
				{
				setState(206); neg_one_reference();
				}
				break;
			case 13:
				{
				setState(207); neg_between();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Pattern_predicateContext extends ParserRuleContext {
		public Negative_predicateContext negative_predicate() {
			return getRuleContext(Negative_predicateContext.class,0);
		}
		public Positive_predicateContext positive_predicate() {
			return getRuleContext(Positive_predicateContext.class,0);
		}
		public Pattern_predicateContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pattern_predicate; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterPattern_predicate(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitPattern_predicate(this);
		}
	}

	public final Pattern_predicateContext pattern_predicate() throws RecognitionException {
		Pattern_predicateContext _localctx = new Pattern_predicateContext(_ctx, getState());
		enterRule(_localctx, 40, RULE_pattern_predicate);
		try {
			setState(212);
			switch (_input.LA(1)) {
			case 10:
				enterOuterAlt(_localctx, 1);
				{
				setState(210); positive_predicate();
				}
				break;
			case 12:
				enterOuterAlt(_localctx, 2);
				{
				setState(211); negative_predicate();
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class DefinitionsContext extends ParserRuleContext {
		public StaticAttr_definitionContext staticAttr_definition(int i) {
			return getRuleContext(StaticAttr_definitionContext.class,i);
		}
		public List<StaticAttr_definitionContext> staticAttr_definition() {
			return getRuleContexts(StaticAttr_definitionContext.class);
		}
		public List<Attr_definitionContext> attr_definition() {
			return getRuleContexts(Attr_definitionContext.class);
		}
		public Attr_definitionContext attr_definition(int i) {
			return getRuleContext(Attr_definitionContext.class,i);
		}
		public DefinitionsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_definitions; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterDefinitions(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitDefinitions(this);
		}
	}

	public final DefinitionsContext definitions() throws RecognitionException {
		DefinitionsContext _localctx = new DefinitionsContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_definitions);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(216);
			switch ( getInterpreter().adaptivePredict(_input,18,_ctx) ) {
			case 1:
				{
				setState(214); staticAttr_definition();
				}
				break;

			case 2:
				{
				setState(215); attr_definition();
				}
				break;
			}
			setState(225);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==5) {
				{
				{
				setState(218); match(5);
				setState(221);
				switch ( getInterpreter().adaptivePredict(_input,19,_ctx) ) {
				case 1:
					{
					setState(219); staticAttr_definition();
					}
					break;

				case 2:
					{
					setState(220); attr_definition();
					}
					break;
				}
				}
				}
				setState(227);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ConsumingContext extends ParserRuleContext {
		public List<TerminalNode> PRED_NAME() { return getTokens(TESLAParser.PRED_NAME); }
		public TerminalNode PRED_NAME(int i) {
			return getToken(TESLAParser.PRED_NAME, i);
		}
		public ConsumingContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_consuming; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterConsuming(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitConsuming(this);
		}
	}

	public final ConsumingContext consuming() throws RecognitionException {
		ConsumingContext _localctx = new ConsumingContext(_ctx, getState());
		enterRule(_localctx, 44, RULE_consuming);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(228); match(PRED_NAME);
			setState(233);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==5) {
				{
				{
				setState(229); match(5);
				setState(230); match(PRED_NAME);
				}
				}
				setState(235);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class PatternContext extends ParserRuleContext {
		public List<Pattern_predicateContext> pattern_predicate() {
			return getRuleContexts(Pattern_predicateContext.class);
		}
		public Pattern_predicateContext pattern_predicate(int i) {
			return getRuleContext(Pattern_predicateContext.class,i);
		}
		public TerminatorContext terminator() {
			return getRuleContext(TerminatorContext.class,0);
		}
		public PatternContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_pattern; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterPattern(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitPattern(this);
		}
	}

	public final PatternContext pattern() throws RecognitionException {
		PatternContext _localctx = new PatternContext(_ctx, getState());
		enterRule(_localctx, 46, RULE_pattern);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(236); terminator();
			setState(240);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==10 || _la==12) {
				{
				{
				setState(237); pattern_predicate();
				}
				}
				setState(242);
				_errHandler.sync(this);
				_la = _input.LA(1);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Ending_ruleContext extends ParserRuleContext {
		public Ending_ruleContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_ending_rule; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterEnding_rule(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitEnding_rule(this);
		}
	}

	public final Ending_ruleContext ending_rule() throws RecognitionException {
		Ending_ruleContext _localctx = new Ending_ruleContext(_ctx, getState());
		enterRule(_localctx, 48, RULE_ending_rule);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(243); match(9);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class Trex_ruleContext extends ParserRuleContext {
		public TerminalNode CONSUMING() { return getToken(TESLAParser.CONSUMING, 0); }
		public TerminalNode WHERE() { return getToken(TESLAParser.WHERE, 0); }
		public TerminalNode DEFINE() { return getToken(TESLAParser.DEFINE, 0); }
		public PatternContext pattern() {
			return getRuleContext(PatternContext.class,0);
		}
		public ConsumingContext consuming() {
			return getRuleContext(ConsumingContext.class,0);
		}
		public DefinitionsContext definitions() {
			return getRuleContext(DefinitionsContext.class,0);
		}
		public Ending_ruleContext ending_rule() {
			return getRuleContext(Ending_ruleContext.class,0);
		}
		public TerminalNode FROM() { return getToken(TESLAParser.FROM, 0); }
		public Ce_definitionContext ce_definition() {
			return getRuleContext(Ce_definitionContext.class,0);
		}
		public Trex_ruleContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_trex_rule; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterTrex_rule(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitTrex_rule(this);
		}
	}

	public final Trex_ruleContext trex_rule() throws RecognitionException {
		Trex_ruleContext _localctx = new Trex_ruleContext(_ctx, getState());
		enterRule(_localctx, 50, RULE_trex_rule);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(245); match(DEFINE);
			setState(246); ce_definition();
			setState(247); match(FROM);
			setState(248); pattern();
			setState(251);
			_la = _input.LA(1);
			if (_la==WHERE) {
				{
				setState(249); match(WHERE);
				setState(250); definitions();
				}
			}

			setState(255);
			_la = _input.LA(1);
			if (_la==CONSUMING) {
				{
				setState(253); match(CONSUMING);
				setState(254); consuming();
				}
			}

			setState(257); ending_rule();
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 7: return expr_sempred((ExprContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0: return precpred(_ctx, 4);

		case 1: return precpred(_ctx, 3);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u0430\ud6d1\u8206\uad2d\u4417\uaef1\u8d80\uaadd\3\"\u0106\4\2\t\2\4"+
		"\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t"+
		"\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\3\2\3\2\3\3\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\5\3\5\3"+
		"\5\5\5D\n\5\3\6\3\6\3\6\3\6\3\6\3\7\3\7\3\7\3\7\3\7\3\b\3\b\3\b\3\b\3"+
		"\b\3\b\5\bV\n\b\3\b\3\b\3\b\5\b[\n\b\7\b]\n\b\f\b\16\b`\13\b\5\bb\n\b"+
		"\3\b\3\b\3\b\3\b\5\bh\n\b\3\t\3\t\3\t\3\t\3\t\3\t\3\t\5\tq\n\t\5\ts\n"+
		"\t\3\t\3\t\3\t\3\t\3\t\3\t\7\t{\n\t\f\t\16\t~\13\t\3\n\3\n\3\n\3\n\3\13"+
		"\3\13\3\13\3\13\3\f\3\f\3\f\3\f\3\r\3\r\3\r\3\r\3\16\3\16\3\16\3\16\3"+
		"\16\3\16\3\16\3\17\3\17\3\17\3\17\3\17\7\17\u009c\n\17\f\17\16\17\u009f"+
		"\13\17\5\17\u00a1\n\17\3\17\3\17\3\20\3\20\3\20\3\20\3\20\5\20\u00aa\n"+
		"\20\3\20\3\20\3\20\3\20\5\20\u00b0\n\20\7\20\u00b2\n\20\f\20\16\20\u00b5"+
		"\13\20\5\20\u00b7\n\20\3\20\3\20\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3"+
		"\22\3\22\3\22\3\23\3\23\3\23\3\23\3\23\3\24\3\24\3\24\3\24\3\24\3\25\3"+
		"\25\3\25\3\25\5\25\u00d3\n\25\3\26\3\26\5\26\u00d7\n\26\3\27\3\27\5\27"+
		"\u00db\n\27\3\27\3\27\3\27\5\27\u00e0\n\27\7\27\u00e2\n\27\f\27\16\27"+
		"\u00e5\13\27\3\30\3\30\3\30\7\30\u00ea\n\30\f\30\16\30\u00ed\13\30\3\31"+
		"\3\31\7\31\u00f1\n\31\f\31\16\31\u00f4\13\31\3\32\3\32\3\33\3\33\3\33"+
		"\3\33\3\33\3\33\5\33\u00fe\n\33\3\33\3\33\5\33\u0102\n\33\3\33\3\33\3"+
		"\33\2\3\20\34\2\4\6\b\n\f\16\20\22\24\26\30\32\34\36 \"$&(*,.\60\62\64"+
		"\2\3\4\2\25\25\34\36\u0107\2\66\3\2\2\2\48\3\2\2\2\6<\3\2\2\2\bC\3\2\2"+
		"\2\nE\3\2\2\2\fJ\3\2\2\2\16O\3\2\2\2\20r\3\2\2\2\22\177\3\2\2\2\24\u0083"+
		"\3\2\2\2\26\u0087\3\2\2\2\30\u008b\3\2\2\2\32\u008f\3\2\2\2\34\u0096\3"+
		"\2\2\2\36\u00a4\3\2\2\2 \u00ba\3\2\2\2\"\u00bc\3\2\2\2$\u00c4\3\2\2\2"+
		"&\u00c9\3\2\2\2(\u00ce\3\2\2\2*\u00d6\3\2\2\2,\u00da\3\2\2\2.\u00e6\3"+
		"\2\2\2\60\u00ee\3\2\2\2\62\u00f5\3\2\2\2\64\u00f7\3\2\2\2\66\67\t\2\2"+
		"\2\67\3\3\2\2\289\7\37\2\29:\7\6\2\2:;\7 \2\2;\5\3\2\2\2<=\7 \2\2=>\7"+
		"\5\2\2>?\7!\2\2?\7\3\2\2\2@D\5\4\3\2AD\7!\2\2BD\5\2\2\2C@\3\2\2\2CA\3"+
		"\2\2\2CB\3\2\2\2D\t\3\2\2\2EF\7\23\2\2FG\7\35\2\2GH\7\21\2\2HI\7\37\2"+
		"\2I\13\3\2\2\2JK\7\17\2\2KL\7\37\2\2LM\7\f\2\2MN\7\37\2\2N\r\3\2\2\2O"+
		"P\7\30\2\2PQ\7\n\2\2QR\5\4\3\2Ra\7\n\2\2SV\5\32\16\2TV\5\30\r\2US\3\2"+
		"\2\2UT\3\2\2\2V^\3\2\2\2WZ\7\7\2\2X[\5\32\16\2Y[\5\30\r\2ZX\3\2\2\2ZY"+
		"\3\2\2\2[]\3\2\2\2\\W\3\2\2\2]`\3\2\2\2^\\\3\2\2\2^_\3\2\2\2_b\3\2\2\2"+
		"`^\3\2\2\2aU\3\2\2\2ab\3\2\2\2bc\3\2\2\2cd\7\4\2\2dg\7\4\2\2eh\5\n\6\2"+
		"fh\5\f\7\2ge\3\2\2\2gf\3\2\2\2h\17\3\2\2\2ij\b\t\1\2jk\7\n\2\2kl\5\20"+
		"\t\2lm\7\4\2\2ms\3\2\2\2nq\5\b\5\2oq\5\16\b\2pn\3\2\2\2po\3\2\2\2qs\3"+
		"\2\2\2ri\3\2\2\2rp\3\2\2\2s|\3\2\2\2tu\f\6\2\2uv\7\32\2\2v{\5\20\t\7w"+
		"x\f\5\2\2xy\7\33\2\2y{\5\20\t\6zt\3\2\2\2zw\3\2\2\2{~\3\2\2\2|z\3\2\2"+
		"\2|}\3\2\2\2}\21\3\2\2\2~|\3\2\2\2\177\u0080\7 \2\2\u0080\u0081\7\t\2"+
		"\2\u0081\u0082\7\26\2\2\u0082\23\3\2\2\2\u0083\u0084\7 \2\2\u0084\u0085"+
		"\7\r\2\2\u0085\u0086\5\2\2\2\u0086\25\3\2\2\2\u0087\u0088\7 \2\2\u0088"+
		"\u0089\7\r\2\2\u0089\u008a\5\20\t\2\u008a\27\3\2\2\2\u008b\u008c\7 \2"+
		"\2\u008c\u008d\7\31\2\2\u008d\u008e\5\2\2\2\u008e\31\3\2\2\2\u008f\u0090"+
		"\7\b\2\2\u0090\u0091\7\26\2\2\u0091\u0092\7\3\2\2\u0092\u0093\7 \2\2\u0093"+
		"\u0094\7\31\2\2\u0094\u0095\5\20\t\2\u0095\33\3\2\2\2\u0096\u0097\7\37"+
		"\2\2\u0097\u00a0\7\n\2\2\u0098\u009d\5\22\n\2\u0099\u009a\7\7\2\2\u009a"+
		"\u009c\5\22\n\2\u009b\u0099\3\2\2\2\u009c\u009f\3\2\2\2\u009d\u009b\3"+
		"\2\2\2\u009d\u009e\3\2\2\2\u009e\u00a1\3\2\2\2\u009f\u009d\3\2\2\2\u00a0"+
		"\u0098\3\2\2\2\u00a0\u00a1\3\2\2\2\u00a1\u00a2\3\2\2\2\u00a2\u00a3\7\4"+
		"\2\2\u00a3\35\3\2\2\2\u00a4\u00a5\7\37\2\2\u00a5\u00b6\7\n\2\2\u00a6\u00aa"+
		"\5\6\4\2\u00a7\u00aa\5\30\r\2\u00a8\u00aa\5\32\16\2\u00a9\u00a6\3\2\2"+
		"\2\u00a9\u00a7\3\2\2\2\u00a9\u00a8\3\2\2\2\u00aa\u00b3\3\2\2\2\u00ab\u00af"+
		"\7\7\2\2\u00ac\u00b0\5\6\4\2\u00ad\u00b0\5\30\r\2\u00ae\u00b0\5\32\16"+
		"\2\u00af\u00ac\3\2\2\2\u00af\u00ad\3\2\2\2\u00af\u00ae\3\2\2\2\u00b0\u00b2"+
		"\3\2\2\2\u00b1\u00ab\3\2\2\2\u00b2\u00b5\3\2\2\2\u00b3\u00b1\3\2\2\2\u00b3"+
		"\u00b4\3\2\2\2\u00b4\u00b7\3\2\2\2\u00b5\u00b3\3\2\2\2\u00b6\u00a9\3\2"+
		"\2\2\u00b6\u00b7\3\2\2\2\u00b7\u00b8\3\2\2\2\u00b8\u00b9\7\4\2\2\u00b9"+
		"\37\3\2\2\2\u00ba\u00bb\5\36\20\2\u00bb!\3\2\2\2\u00bc\u00bd\7\f\2\2\u00bd"+
		"\u00be\7\27\2\2\u00be\u00bf\5\36\20\2\u00bf\u00c0\7\23\2\2\u00c0\u00c1"+
		"\7\35\2\2\u00c1\u00c2\7\21\2\2\u00c2\u00c3\7\37\2\2\u00c3#\3\2\2\2\u00c4"+
		"\u00c5\7\23\2\2\u00c5\u00c6\7\35\2\2\u00c6\u00c7\7\21\2\2\u00c7\u00c8"+
		"\7\37\2\2\u00c8%\3\2\2\2\u00c9\u00ca\7\17\2\2\u00ca\u00cb\7\37\2\2\u00cb"+
		"\u00cc\7\f\2\2\u00cc\u00cd\7\37\2\2\u00cd\'\3\2\2\2\u00ce\u00cf\7\16\2"+
		"\2\u00cf\u00d2\5\36\20\2\u00d0\u00d3\5$\23\2\u00d1\u00d3\5&\24\2\u00d2"+
		"\u00d0\3\2\2\2\u00d2\u00d1\3\2\2\2\u00d3)\3\2\2\2\u00d4\u00d7\5\"\22\2"+
		"\u00d5\u00d7\5(\25\2\u00d6\u00d4\3\2\2\2\u00d6\u00d5\3\2\2\2\u00d7+\3"+
		"\2\2\2\u00d8\u00db\5\24\13\2\u00d9\u00db\5\26\f\2\u00da\u00d8\3\2\2\2"+
		"\u00da\u00d9\3\2\2\2\u00db\u00e3\3\2\2\2\u00dc\u00df\7\7\2\2\u00dd\u00e0"+
		"\5\24\13\2\u00de\u00e0\5\26\f\2\u00df\u00dd\3\2\2\2\u00df\u00de\3\2\2"+
		"\2\u00e0\u00e2\3\2\2\2\u00e1\u00dc\3\2\2\2\u00e2\u00e5\3\2\2\2\u00e3\u00e1"+
		"\3\2\2\2\u00e3\u00e4\3\2\2\2\u00e4-\3\2\2\2\u00e5\u00e3\3\2\2\2\u00e6"+
		"\u00eb\7\37\2\2\u00e7\u00e8\7\7\2\2\u00e8\u00ea\7\37\2\2\u00e9\u00e7\3"+
		"\2\2\2\u00ea\u00ed\3\2\2\2\u00eb\u00e9\3\2\2\2\u00eb\u00ec\3\2\2\2\u00ec"+
		"/\3\2\2\2\u00ed\u00eb\3\2\2\2\u00ee\u00f2\5 \21\2\u00ef\u00f1\5*\26\2"+
		"\u00f0\u00ef\3\2\2\2\u00f1\u00f4\3\2\2\2\u00f2\u00f0\3\2\2\2\u00f2\u00f3"+
		"\3\2\2\2\u00f3\61\3\2\2\2\u00f4\u00f2\3\2\2\2\u00f5\u00f6\7\13\2\2\u00f6"+
		"\63\3\2\2\2\u00f7\u00f8\7\20\2\2\u00f8\u00f9\5\34\17\2\u00f9\u00fa\7\21"+
		"\2\2\u00fa\u00fd\5\60\31\2\u00fb\u00fc\7\22\2\2\u00fc\u00fe\5,\27\2\u00fd"+
		"\u00fb\3\2\2\2\u00fd\u00fe\3\2\2\2\u00fe\u0101\3\2\2\2\u00ff\u0100\7\24"+
		"\2\2\u0100\u0102\5.\30\2\u0101\u00ff\3\2\2\2\u0101\u0102\3\2\2\2\u0102"+
		"\u0103\3\2\2\2\u0103\u0104\5\62\32\2\u0104\65\3\2\2\2\33CUZ^agprz|\u009d"+
		"\u00a0\u00a9\u00af\u00b3\u00b6\u00d2\u00d6\u00da\u00df\u00e3\u00eb\u00f2"+
		"\u00fd\u0101";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}