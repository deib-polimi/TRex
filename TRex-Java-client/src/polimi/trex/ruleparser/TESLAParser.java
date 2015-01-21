// Generated from TESLA.g4 by ANTLR 4.2.2
package polimi.trex.ruleparser;
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
		T__15=1, T__14=2, T__13=3, T__12=4, T__11=5, T__10=6, T__9=7, T__8=8, 
		T__7=9, T__6=10, T__5=11, T__4=12, T__3=13, T__2=14, T__1=15, T__0=16, 
		ASSIGN=17, DEFINE=18, FROM=19, WHERE=20, CONSUMING=21, VALTYPE=22, SEL_POLICY=23, 
		AGGR_FUN=24, OPERATOR=25, BINOP_MUL=26, BINOP_ADD=27, INT_VAL=28, FLOAT_VAL=29, 
		BOOL_VAL=30, STRING_VAL=31, EVT_NAME=32, ATTR_NAME=33, PARAM_NAME=34, 
		WS=35;
	public static final String[] tokenNames = {
		"<INVALID>", "'from'", "']'", "')'", "'=>'", "'.'", "','", "'['", "':'", 
		"'('", "'not'", "';'", "'within'", "'as'", "'and'", "':='", "'between'", 
		"'Assign'", "'Define'", "'From'", "'Where'", "'Consuming'", "VALTYPE", 
		"SEL_POLICY", "AGGR_FUN", "OPERATOR", "BINOP_MUL", "BINOP_ADD", "INT_VAL", 
		"FLOAT_VAL", "BOOL_VAL", "STRING_VAL", "EVT_NAME", "ATTR_NAME", "PARAM_NAME", 
		"WS"
	};
	public static final int
		RULE_static_reference = 0, RULE_packet_reference = 1, RULE_param_mapping = 2, 
		RULE_param_atom = 3, RULE_agg_one_reference = 4, RULE_agg_between = 5, 
		RULE_aggregate_atom = 6, RULE_expr = 7, RULE_attr_declaration = 8, RULE_staticAttr_definition = 9, 
		RULE_attr_definition = 10, RULE_attr_constraint = 11, RULE_attr_parameter = 12, 
		RULE_predicate = 13, RULE_event_alias = 14, RULE_terminator = 15, RULE_positive_predicate = 16, 
		RULE_neg_one_reference = 17, RULE_neg_between = 18, RULE_negative_predicate = 19, 
		RULE_pattern_predicate = 20, RULE_event_declaration = 21, RULE_event_declarations = 22, 
		RULE_ce_definition = 23, RULE_pattern = 24, RULE_definitions = 25, RULE_consuming = 26, 
		RULE_ending_rule = 27, RULE_trex_rule = 28;
	public static final String[] ruleNames = {
		"static_reference", "packet_reference", "param_mapping", "param_atom", 
		"agg_one_reference", "agg_between", "aggregate_atom", "expr", "attr_declaration", 
		"staticAttr_definition", "attr_definition", "attr_constraint", "attr_parameter", 
		"predicate", "event_alias", "terminator", "positive_predicate", "neg_one_reference", 
		"neg_between", "negative_predicate", "pattern_predicate", "event_declaration", 
		"event_declarations", "ce_definition", "pattern", "definitions", "consuming", 
		"ending_rule", "trex_rule"
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


	StringBuilder buf = new StringBuilder(); // can't make locals in lexer rules

	public TESLAParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}
	public static class Static_referenceContext extends ParserRuleContext {
		public TerminalNode STRING_VAL() { return getToken(TESLAParser.STRING_VAL, 0); }
		public TerminalNode BOOL_VAL() { return getToken(TESLAParser.BOOL_VAL, 0); }
		public TerminalNode FLOAT_VAL() { return getToken(TESLAParser.FLOAT_VAL, 0); }
		public TerminalNode INT_VAL() { return getToken(TESLAParser.INT_VAL, 0); }
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
			setState(58);
			_la = _input.LA(1);
			if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << INT_VAL) | (1L << FLOAT_VAL) | (1L << BOOL_VAL) | (1L << STRING_VAL))) != 0)) ) {
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
		public TerminalNode ATTR_NAME() { return getToken(TESLAParser.ATTR_NAME, 0); }
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
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
			setState(60); match(EVT_NAME);
			setState(61); match(5);
			setState(62); match(ATTR_NAME);
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
			setState(64); match(ATTR_NAME);
			setState(65); match(4);
			setState(66); match(PARAM_NAME);
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
			setState(71);
			switch (_input.LA(1)) {
			case EVT_NAME:
				{
				setState(68); packet_reference();
				}
				break;
			case PARAM_NAME:
				{
				setState(69); match(PARAM_NAME);
				}
				break;
			case INT_VAL:
			case FLOAT_VAL:
			case BOOL_VAL:
			case STRING_VAL:
				{
				setState(70); static_reference();
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
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
		public TerminalNode INT_VAL() { return getToken(TESLAParser.INT_VAL, 0); }
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
			setState(73); match(12);
			setState(74); match(INT_VAL);
			setState(75); match(1);
			setState(76); match(EVT_NAME);
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
		public TerminalNode EVT_NAME(int i) {
			return getToken(TESLAParser.EVT_NAME, i);
		}
		public List<TerminalNode> EVT_NAME() { return getTokens(TESLAParser.EVT_NAME); }
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
			setState(78); match(16);
			setState(79); match(EVT_NAME);
			setState(80); match(14);
			setState(81); match(EVT_NAME);
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
			setState(83); match(AGGR_FUN);
			setState(84); match(9);
			setState(85); packet_reference();
			setState(86); match(9);
			setState(101);
			_la = _input.LA(1);
			if (_la==7 || _la==ATTR_NAME) {
				{
				setState(89);
				switch (_input.LA(1)) {
				case 7:
					{
					setState(87); attr_parameter();
					}
					break;
				case ATTR_NAME:
					{
					setState(88); attr_constraint();
					}
					break;
				default:
					throw new NoViableAltException(this);
				}
				setState(98);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==6) {
					{
					{
					setState(91); match(6);
					setState(94);
					switch (_input.LA(1)) {
					case 7:
						{
						setState(92); attr_parameter();
						}
						break;
					case ATTR_NAME:
						{
						setState(93); attr_constraint();
						}
						break;
					default:
						throw new NoViableAltException(this);
					}
					}
					}
					setState(100);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(103); match(3);
			setState(104); match(3);
			setState(107);
			switch (_input.LA(1)) {
			case 12:
				{
				setState(105); agg_one_reference();
				}
				break;
			case 16:
				{
				setState(106); agg_between();
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
			setState(118);
			switch (_input.LA(1)) {
			case 9:
				{
				setState(110); match(9);
				setState(111); expr(0);
				setState(112); match(3);
				}
				break;
			case AGGR_FUN:
			case INT_VAL:
			case FLOAT_VAL:
			case BOOL_VAL:
			case STRING_VAL:
			case EVT_NAME:
			case PARAM_NAME:
				{
				setState(116);
				switch (_input.LA(1)) {
				case INT_VAL:
				case FLOAT_VAL:
				case BOOL_VAL:
				case STRING_VAL:
				case EVT_NAME:
				case PARAM_NAME:
					{
					setState(114); param_atom();
					}
					break;
				case AGGR_FUN:
					{
					setState(115); aggregate_atom();
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
			setState(128);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			while ( _alt!=2 && _alt!=ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(126);
					switch ( getInterpreter().adaptivePredict(_input,8,_ctx) ) {
					case 1:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(120);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(121); match(BINOP_MUL);
						setState(122); expr(5);
						}
						break;

					case 2:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(123);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(124); match(BINOP_ADD);
						setState(125); expr(4);
						}
						break;
					}
					} 
				}
				setState(130);
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
			setState(131); match(ATTR_NAME);
			setState(132); match(8);
			setState(133); match(VALTYPE);
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
			setState(135); match(ATTR_NAME);
			setState(136); match(15);
			setState(137); static_reference();
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
			setState(139); match(ATTR_NAME);
			setState(140); match(15);
			setState(141); expr(0);
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
			setState(143); match(ATTR_NAME);
			setState(144); match(OPERATOR);
			setState(145); static_reference();
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
			setState(147); match(7);
			setState(148); match(VALTYPE);
			setState(149); match(2);
			setState(150); match(ATTR_NAME);
			setState(151); match(OPERATOR);
			setState(152); expr(0);
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
		public Event_aliasContext event_alias() {
			return getRuleContext(Event_aliasContext.class,0);
		}
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
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
		enterRule(_localctx, 26, RULE_predicate);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(154); match(EVT_NAME);
			setState(155); match(9);
			setState(172);
			_la = _input.LA(1);
			if (_la==7 || _la==ATTR_NAME) {
				{
				setState(159);
				switch ( getInterpreter().adaptivePredict(_input,10,_ctx) ) {
				case 1:
					{
					setState(156); param_mapping();
					}
					break;

				case 2:
					{
					setState(157); attr_constraint();
					}
					break;

				case 3:
					{
					setState(158); attr_parameter();
					}
					break;
				}
				setState(169);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==6) {
					{
					{
					setState(161); match(6);
					setState(165);
					switch ( getInterpreter().adaptivePredict(_input,11,_ctx) ) {
					case 1:
						{
						setState(162); param_mapping();
						}
						break;

					case 2:
						{
						setState(163); attr_constraint();
						}
						break;

					case 3:
						{
						setState(164); attr_parameter();
						}
						break;
					}
					}
					}
					setState(171);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(174); match(3);
			setState(176);
			_la = _input.LA(1);
			if (_la==13) {
				{
				setState(175); event_alias();
				}
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

	public static class Event_aliasContext extends ParserRuleContext {
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
		public Event_aliasContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_event_alias; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterEvent_alias(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitEvent_alias(this);
		}
	}

	public final Event_aliasContext event_alias() throws RecognitionException {
		Event_aliasContext _localctx = new Event_aliasContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_event_alias);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(178); match(13);
			setState(179); match(EVT_NAME);
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
			setState(181); predicate();
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
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
		public PredicateContext predicate() {
			return getRuleContext(PredicateContext.class,0);
		}
		public TerminalNode INT_VAL() { return getToken(TESLAParser.INT_VAL, 0); }
		public TerminalNode SEL_POLICY() { return getToken(TESLAParser.SEL_POLICY, 0); }
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
			setState(183); match(14);
			setState(184); match(SEL_POLICY);
			setState(185); predicate();
			setState(186); match(12);
			setState(187); match(INT_VAL);
			setState(188); match(1);
			setState(189); match(EVT_NAME);
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
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
		public TerminalNode INT_VAL() { return getToken(TESLAParser.INT_VAL, 0); }
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
			setState(191); match(12);
			setState(192); match(INT_VAL);
			setState(193); match(1);
			setState(194); match(EVT_NAME);
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
		public TerminalNode EVT_NAME(int i) {
			return getToken(TESLAParser.EVT_NAME, i);
		}
		public List<TerminalNode> EVT_NAME() { return getTokens(TESLAParser.EVT_NAME); }
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
			setState(196); match(16);
			setState(197); match(EVT_NAME);
			setState(198); match(14);
			setState(199); match(EVT_NAME);
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
			setState(201); match(14);
			setState(202); match(10);
			setState(203); predicate();
			setState(206);
			switch (_input.LA(1)) {
			case 12:
				{
				setState(204); neg_one_reference();
				}
				break;
			case 16:
				{
				setState(205); neg_between();
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
			setState(210);
			switch ( getInterpreter().adaptivePredict(_input,16,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(208); positive_predicate();
				}
				break;

			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(209); negative_predicate();
				}
				break;
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

	public static class Event_declarationContext extends ParserRuleContext {
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
		public TerminalNode INT_VAL() { return getToken(TESLAParser.INT_VAL, 0); }
		public Event_declarationContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_event_declaration; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterEvent_declaration(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitEvent_declaration(this);
		}
	}

	public final Event_declarationContext event_declaration() throws RecognitionException {
		Event_declarationContext _localctx = new Event_declarationContext(_ctx, getState());
		enterRule(_localctx, 42, RULE_event_declaration);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(212); match(INT_VAL);
			setState(213); match(4);
			setState(214); match(EVT_NAME);
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

	public static class Event_declarationsContext extends ParserRuleContext {
		public List<Event_declarationContext> event_declaration() {
			return getRuleContexts(Event_declarationContext.class);
		}
		public Event_declarationContext event_declaration(int i) {
			return getRuleContext(Event_declarationContext.class,i);
		}
		public Event_declarationsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_event_declarations; }
		@Override
		public void enterRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).enterEvent_declarations(this);
		}
		@Override
		public void exitRule(ParseTreeListener listener) {
			if ( listener instanceof TESLAListener ) ((TESLAListener)listener).exitEvent_declarations(this);
		}
	}

	public final Event_declarationsContext event_declarations() throws RecognitionException {
		Event_declarationsContext _localctx = new Event_declarationsContext(_ctx, getState());
		enterRule(_localctx, 44, RULE_event_declarations);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(216); event_declaration();
			setState(221);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==6) {
				{
				{
				setState(217); match(6);
				setState(218); event_declaration();
				}
				}
				setState(223);
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

	public static class Ce_definitionContext extends ParserRuleContext {
		public TerminalNode EVT_NAME() { return getToken(TESLAParser.EVT_NAME, 0); }
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
		enterRule(_localctx, 46, RULE_ce_definition);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(224); match(EVT_NAME);
			setState(225); match(9);
			setState(234);
			_la = _input.LA(1);
			if (_la==ATTR_NAME) {
				{
				setState(226); attr_declaration();
				setState(231);
				_errHandler.sync(this);
				_la = _input.LA(1);
				while (_la==6) {
					{
					{
					setState(227); match(6);
					setState(228); attr_declaration();
					}
					}
					setState(233);
					_errHandler.sync(this);
					_la = _input.LA(1);
				}
				}
			}

			setState(236); match(3);
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
		enterRule(_localctx, 48, RULE_pattern);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(238); terminator();
			setState(242);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==14) {
				{
				{
				setState(239); pattern_predicate();
				}
				}
				setState(244);
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
		enterRule(_localctx, 50, RULE_definitions);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(247);
			switch ( getInterpreter().adaptivePredict(_input,21,_ctx) ) {
			case 1:
				{
				setState(245); staticAttr_definition();
				}
				break;

			case 2:
				{
				setState(246); attr_definition();
				}
				break;
			}
			setState(256);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==6) {
				{
				{
				setState(249); match(6);
				setState(252);
				switch ( getInterpreter().adaptivePredict(_input,22,_ctx) ) {
				case 1:
					{
					setState(250); staticAttr_definition();
					}
					break;

				case 2:
					{
					setState(251); attr_definition();
					}
					break;
				}
				}
				}
				setState(258);
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
		public TerminalNode EVT_NAME(int i) {
			return getToken(TESLAParser.EVT_NAME, i);
		}
		public List<TerminalNode> EVT_NAME() { return getTokens(TESLAParser.EVT_NAME); }
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
		enterRule(_localctx, 52, RULE_consuming);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(259); match(EVT_NAME);
			setState(264);
			_errHandler.sync(this);
			_la = _input.LA(1);
			while (_la==6) {
				{
				{
				setState(260); match(6);
				setState(261); match(EVT_NAME);
				}
				}
				setState(266);
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
		enterRule(_localctx, 54, RULE_ending_rule);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(267); match(11);
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
		public Event_declarationsContext event_declarations() {
			return getRuleContext(Event_declarationsContext.class,0);
		}
		public TerminalNode ASSIGN() { return getToken(TESLAParser.ASSIGN, 0); }
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
		enterRule(_localctx, 56, RULE_trex_rule);
		int _la;
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(269); match(ASSIGN);
			setState(270); event_declarations();
			setState(271); match(DEFINE);
			setState(272); ce_definition();
			setState(273); match(FROM);
			setState(274); pattern();
			setState(277);
			_la = _input.LA(1);
			if (_la==WHERE) {
				{
				setState(275); match(WHERE);
				setState(276); definitions();
				}
			}

			setState(281);
			_la = _input.LA(1);
			if (_la==CONSUMING) {
				{
				setState(279); match(CONSUMING);
				setState(280); consuming();
				}
			}

			setState(283); ending_rule();
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
		"\3\u0430\ud6d1\u8206\uad2d\u4417\uaef1\u8d80\uaadd\3%\u0120\4\2\t\2\4"+
		"\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t"+
		"\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\3\2\3\2\3\3\3\3\3\3"+
		"\3\3\3\4\3\4\3\4\3\4\3\5\3\5\3\5\5\5J\n\5\3\6\3\6\3\6\3\6\3\6\3\7\3\7"+
		"\3\7\3\7\3\7\3\b\3\b\3\b\3\b\3\b\3\b\5\b\\\n\b\3\b\3\b\3\b\5\ba\n\b\7"+
		"\bc\n\b\f\b\16\bf\13\b\5\bh\n\b\3\b\3\b\3\b\3\b\5\bn\n\b\3\t\3\t\3\t\3"+
		"\t\3\t\3\t\3\t\5\tw\n\t\5\ty\n\t\3\t\3\t\3\t\3\t\3\t\3\t\7\t\u0081\n\t"+
		"\f\t\16\t\u0084\13\t\3\n\3\n\3\n\3\n\3\13\3\13\3\13\3\13\3\f\3\f\3\f\3"+
		"\f\3\r\3\r\3\r\3\r\3\16\3\16\3\16\3\16\3\16\3\16\3\16\3\17\3\17\3\17\3"+
		"\17\3\17\5\17\u00a2\n\17\3\17\3\17\3\17\3\17\5\17\u00a8\n\17\7\17\u00aa"+
		"\n\17\f\17\16\17\u00ad\13\17\5\17\u00af\n\17\3\17\3\17\5\17\u00b3\n\17"+
		"\3\20\3\20\3\20\3\21\3\21\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\23"+
		"\3\23\3\23\3\23\3\23\3\24\3\24\3\24\3\24\3\24\3\25\3\25\3\25\3\25\3\25"+
		"\5\25\u00d1\n\25\3\26\3\26\5\26\u00d5\n\26\3\27\3\27\3\27\3\27\3\30\3"+
		"\30\3\30\7\30\u00de\n\30\f\30\16\30\u00e1\13\30\3\31\3\31\3\31\3\31\3"+
		"\31\7\31\u00e8\n\31\f\31\16\31\u00eb\13\31\5\31\u00ed\n\31\3\31\3\31\3"+
		"\32\3\32\7\32\u00f3\n\32\f\32\16\32\u00f6\13\32\3\33\3\33\5\33\u00fa\n"+
		"\33\3\33\3\33\3\33\5\33\u00ff\n\33\7\33\u0101\n\33\f\33\16\33\u0104\13"+
		"\33\3\34\3\34\3\34\7\34\u0109\n\34\f\34\16\34\u010c\13\34\3\35\3\35\3"+
		"\36\3\36\3\36\3\36\3\36\3\36\3\36\3\36\5\36\u0118\n\36\3\36\3\36\5\36"+
		"\u011c\n\36\3\36\3\36\3\36\2\3\20\37\2\4\6\b\n\f\16\20\22\24\26\30\32"+
		"\34\36 \"$&(*,.\60\62\64\668:\2\3\3\2\36!\u0120\2<\3\2\2\2\4>\3\2\2\2"+
		"\6B\3\2\2\2\bI\3\2\2\2\nK\3\2\2\2\fP\3\2\2\2\16U\3\2\2\2\20x\3\2\2\2\22"+
		"\u0085\3\2\2\2\24\u0089\3\2\2\2\26\u008d\3\2\2\2\30\u0091\3\2\2\2\32\u0095"+
		"\3\2\2\2\34\u009c\3\2\2\2\36\u00b4\3\2\2\2 \u00b7\3\2\2\2\"\u00b9\3\2"+
		"\2\2$\u00c1\3\2\2\2&\u00c6\3\2\2\2(\u00cb\3\2\2\2*\u00d4\3\2\2\2,\u00d6"+
		"\3\2\2\2.\u00da\3\2\2\2\60\u00e2\3\2\2\2\62\u00f0\3\2\2\2\64\u00f9\3\2"+
		"\2\2\66\u0105\3\2\2\28\u010d\3\2\2\2:\u010f\3\2\2\2<=\t\2\2\2=\3\3\2\2"+
		"\2>?\7\"\2\2?@\7\7\2\2@A\7#\2\2A\5\3\2\2\2BC\7#\2\2CD\7\6\2\2DE\7$\2\2"+
		"E\7\3\2\2\2FJ\5\4\3\2GJ\7$\2\2HJ\5\2\2\2IF\3\2\2\2IG\3\2\2\2IH\3\2\2\2"+
		"J\t\3\2\2\2KL\7\16\2\2LM\7\36\2\2MN\7\3\2\2NO\7\"\2\2O\13\3\2\2\2PQ\7"+
		"\22\2\2QR\7\"\2\2RS\7\20\2\2ST\7\"\2\2T\r\3\2\2\2UV\7\32\2\2VW\7\13\2"+
		"\2WX\5\4\3\2Xg\7\13\2\2Y\\\5\32\16\2Z\\\5\30\r\2[Y\3\2\2\2[Z\3\2\2\2\\"+
		"d\3\2\2\2]`\7\b\2\2^a\5\32\16\2_a\5\30\r\2`^\3\2\2\2`_\3\2\2\2ac\3\2\2"+
		"\2b]\3\2\2\2cf\3\2\2\2db\3\2\2\2de\3\2\2\2eh\3\2\2\2fd\3\2\2\2g[\3\2\2"+
		"\2gh\3\2\2\2hi\3\2\2\2ij\7\5\2\2jm\7\5\2\2kn\5\n\6\2ln\5\f\7\2mk\3\2\2"+
		"\2ml\3\2\2\2n\17\3\2\2\2op\b\t\1\2pq\7\13\2\2qr\5\20\t\2rs\7\5\2\2sy\3"+
		"\2\2\2tw\5\b\5\2uw\5\16\b\2vt\3\2\2\2vu\3\2\2\2wy\3\2\2\2xo\3\2\2\2xv"+
		"\3\2\2\2y\u0082\3\2\2\2z{\f\6\2\2{|\7\34\2\2|\u0081\5\20\t\7}~\f\5\2\2"+
		"~\177\7\35\2\2\177\u0081\5\20\t\6\u0080z\3\2\2\2\u0080}\3\2\2\2\u0081"+
		"\u0084\3\2\2\2\u0082\u0080\3\2\2\2\u0082\u0083\3\2\2\2\u0083\21\3\2\2"+
		"\2\u0084\u0082\3\2\2\2\u0085\u0086\7#\2\2\u0086\u0087\7\n\2\2\u0087\u0088"+
		"\7\30\2\2\u0088\23\3\2\2\2\u0089\u008a\7#\2\2\u008a\u008b\7\21\2\2\u008b"+
		"\u008c\5\2\2\2\u008c\25\3\2\2\2\u008d\u008e\7#\2\2\u008e\u008f\7\21\2"+
		"\2\u008f\u0090\5\20\t\2\u0090\27\3\2\2\2\u0091\u0092\7#\2\2\u0092\u0093"+
		"\7\33\2\2\u0093\u0094\5\2\2\2\u0094\31\3\2\2\2\u0095\u0096\7\t\2\2\u0096"+
		"\u0097\7\30\2\2\u0097\u0098\7\4\2\2\u0098\u0099\7#\2\2\u0099\u009a\7\33"+
		"\2\2\u009a\u009b\5\20\t\2\u009b\33\3\2\2\2\u009c\u009d\7\"\2\2\u009d\u00ae"+
		"\7\13\2\2\u009e\u00a2\5\6\4\2\u009f\u00a2\5\30\r\2\u00a0\u00a2\5\32\16"+
		"\2\u00a1\u009e\3\2\2\2\u00a1\u009f\3\2\2\2\u00a1\u00a0\3\2\2\2\u00a2\u00ab"+
		"\3\2\2\2\u00a3\u00a7\7\b\2\2\u00a4\u00a8\5\6\4\2\u00a5\u00a8\5\30\r\2"+
		"\u00a6\u00a8\5\32\16\2\u00a7\u00a4\3\2\2\2\u00a7\u00a5\3\2\2\2\u00a7\u00a6"+
		"\3\2\2\2\u00a8\u00aa\3\2\2\2\u00a9\u00a3\3\2\2\2\u00aa\u00ad\3\2\2\2\u00ab"+
		"\u00a9\3\2\2\2\u00ab\u00ac\3\2\2\2\u00ac\u00af\3\2\2\2\u00ad\u00ab\3\2"+
		"\2\2\u00ae\u00a1\3\2\2\2\u00ae\u00af\3\2\2\2\u00af\u00b0\3\2\2\2\u00b0"+
		"\u00b2\7\5\2\2\u00b1\u00b3\5\36\20\2\u00b2\u00b1\3\2\2\2\u00b2\u00b3\3"+
		"\2\2\2\u00b3\35\3\2\2\2\u00b4\u00b5\7\17\2\2\u00b5\u00b6\7\"\2\2\u00b6"+
		"\37\3\2\2\2\u00b7\u00b8\5\34\17\2\u00b8!\3\2\2\2\u00b9\u00ba\7\20\2\2"+
		"\u00ba\u00bb\7\31\2\2\u00bb\u00bc\5\34\17\2\u00bc\u00bd\7\16\2\2\u00bd"+
		"\u00be\7\36\2\2\u00be\u00bf\7\3\2\2\u00bf\u00c0\7\"\2\2\u00c0#\3\2\2\2"+
		"\u00c1\u00c2\7\16\2\2\u00c2\u00c3\7\36\2\2\u00c3\u00c4\7\3\2\2\u00c4\u00c5"+
		"\7\"\2\2\u00c5%\3\2\2\2\u00c6\u00c7\7\22\2\2\u00c7\u00c8\7\"\2\2\u00c8"+
		"\u00c9\7\20\2\2\u00c9\u00ca\7\"\2\2\u00ca\'\3\2\2\2\u00cb\u00cc\7\20\2"+
		"\2\u00cc\u00cd\7\f\2\2\u00cd\u00d0\5\34\17\2\u00ce\u00d1\5$\23\2\u00cf"+
		"\u00d1\5&\24\2\u00d0\u00ce\3\2\2\2\u00d0\u00cf\3\2\2\2\u00d1)\3\2\2\2"+
		"\u00d2\u00d5\5\"\22\2\u00d3\u00d5\5(\25\2\u00d4\u00d2\3\2\2\2\u00d4\u00d3"+
		"\3\2\2\2\u00d5+\3\2\2\2\u00d6\u00d7\7\36\2\2\u00d7\u00d8\7\6\2\2\u00d8"+
		"\u00d9\7\"\2\2\u00d9-\3\2\2\2\u00da\u00df\5,\27\2\u00db\u00dc\7\b\2\2"+
		"\u00dc\u00de\5,\27\2\u00dd\u00db\3\2\2\2\u00de\u00e1\3\2\2\2\u00df\u00dd"+
		"\3\2\2\2\u00df\u00e0\3\2\2\2\u00e0/\3\2\2\2\u00e1\u00df\3\2\2\2\u00e2"+
		"\u00e3\7\"\2\2\u00e3\u00ec\7\13\2\2\u00e4\u00e9\5\22\n\2\u00e5\u00e6\7"+
		"\b\2\2\u00e6\u00e8\5\22\n\2\u00e7\u00e5\3\2\2\2\u00e8\u00eb\3\2\2\2\u00e9"+
		"\u00e7\3\2\2\2\u00e9\u00ea\3\2\2\2\u00ea\u00ed\3\2\2\2\u00eb\u00e9\3\2"+
		"\2\2\u00ec\u00e4\3\2\2\2\u00ec\u00ed\3\2\2\2\u00ed\u00ee\3\2\2\2\u00ee"+
		"\u00ef\7\5\2\2\u00ef\61\3\2\2\2\u00f0\u00f4\5 \21\2\u00f1\u00f3\5*\26"+
		"\2\u00f2\u00f1\3\2\2\2\u00f3\u00f6\3\2\2\2\u00f4\u00f2\3\2\2\2\u00f4\u00f5"+
		"\3\2\2\2\u00f5\63\3\2\2\2\u00f6\u00f4\3\2\2\2\u00f7\u00fa\5\24\13\2\u00f8"+
		"\u00fa\5\26\f\2\u00f9\u00f7\3\2\2\2\u00f9\u00f8\3\2\2\2\u00fa\u0102\3"+
		"\2\2\2\u00fb\u00fe\7\b\2\2\u00fc\u00ff\5\24\13\2\u00fd\u00ff\5\26\f\2"+
		"\u00fe\u00fc\3\2\2\2\u00fe\u00fd\3\2\2\2\u00ff\u0101\3\2\2\2\u0100\u00fb"+
		"\3\2\2\2\u0101\u0104\3\2\2\2\u0102\u0100\3\2\2\2\u0102\u0103\3\2\2\2\u0103"+
		"\65\3\2\2\2\u0104\u0102\3\2\2\2\u0105\u010a\7\"\2\2\u0106\u0107\7\b\2"+
		"\2\u0107\u0109\7\"\2\2\u0108\u0106\3\2\2\2\u0109\u010c\3\2\2\2\u010a\u0108"+
		"\3\2\2\2\u010a\u010b\3\2\2\2\u010b\67\3\2\2\2\u010c\u010a\3\2\2\2\u010d"+
		"\u010e\7\r\2\2\u010e9\3\2\2\2\u010f\u0110\7\23\2\2\u0110\u0111\5.\30\2"+
		"\u0111\u0112\7\24\2\2\u0112\u0113\5\60\31\2\u0113\u0114\7\25\2\2\u0114"+
		"\u0117\5\62\32\2\u0115\u0116\7\26\2\2\u0116\u0118\5\64\33\2\u0117\u0115"+
		"\3\2\2\2\u0117\u0118\3\2\2\2\u0118\u011b\3\2\2\2\u0119\u011a\7\27\2\2"+
		"\u011a\u011c\5\66\34\2\u011b\u0119\3\2\2\2\u011b\u011c\3\2\2\2\u011c\u011d"+
		"\3\2\2\2\u011d\u011e\58\35\2\u011e;\3\2\2\2\35I[`dgmvx\u0080\u0082\u00a1"+
		"\u00a7\u00ab\u00ae\u00b2\u00d0\u00d4\u00df\u00e9\u00ec\u00f4\u00f9\u00fe"+
		"\u0102\u010a\u0117\u011b";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}