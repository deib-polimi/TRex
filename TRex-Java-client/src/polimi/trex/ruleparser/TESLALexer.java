// Generated from TESLA.g4 by ANTLR 4.2.2
package polimi.trex.ruleparser;
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class TESLALexer extends Lexer {
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
	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	public static final String[] tokenNames = {
		"<INVALID>",
		"'from'", "']'", "')'", "'=>'", "'.'", "','", "'['", "':'", "'('", "'not'", 
		"';'", "'within'", "'as'", "'and'", "':='", "'between'", "'Assign'", "'Define'", 
		"'From'", "'Where'", "'Consuming'", "VALTYPE", "SEL_POLICY", "AGGR_FUN", 
		"OPERATOR", "BINOP_MUL", "BINOP_ADD", "INT_VAL", "FLOAT_VAL", "BOOL_VAL", 
		"STRING_VAL", "EVT_NAME", "ATTR_NAME", "PARAM_NAME", "WS"
	};
	public static final String[] ruleNames = {
		"T__15", "T__14", "T__13", "T__12", "T__11", "T__10", "T__9", "T__8", 
		"T__7", "T__6", "T__5", "T__4", "T__3", "T__2", "T__1", "T__0", "ASSIGN", 
		"DEFINE", "FROM", "WHERE", "CONSUMING", "VALTYPE", "SEL_POLICY", "AGGR_FUN", 
		"OPERATOR", "BINOP_MUL", "BINOP_ADD", "INT_VAL", "FLOAT_VAL", "BOOL_VAL", 
		"STRING_VAL", "EVT_NAME", "ATTR_NAME", "PARAM_NAME", "WS"
	};


	StringBuilder buf = new StringBuilder(); // can't make locals in lexer rules


	public TESLALexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "TESLA.g4"; }

	@Override
	public String[] getTokenNames() { return tokenNames; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	@Override
	public void action(RuleContext _localctx, int ruleIndex, int actionIndex) {
		switch (ruleIndex) {
		case 30: STRING_VAL_action((RuleContext)_localctx, actionIndex); break;
		}
	}
	private void STRING_VAL_action(RuleContext _localctx, int actionIndex) {
		switch (actionIndex) {
		case 0: buf.append('\r'); break;

		case 1: buf.append('\n'); break;

		case 2: buf.append('\t'); break;

		case 3: buf.append('\\'); break;

		case 4: buf.append('"'); break;

		case 5: buf.append((char)_input.LA(-1)); break;

		case 6: setText(buf.toString()); buf.setLength(0); System.out.println(getText()); break;
		}
	}

	public static final String _serializedATN =
		"\3\u0430\ud6d1\u8206\uad2d\u4417\uaef1\u8d80\uaadd\2%\u0131\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\4#\t#\4$\t$\3\2\3\2\3\2\3\2\3\2\3\3\3\3\3\4\3\4\3\5\3\5\3"+
		"\5\3\6\3\6\3\7\3\7\3\b\3\b\3\t\3\t\3\n\3\n\3\13\3\13\3\13\3\13\3\f\3\f"+
		"\3\r\3\r\3\r\3\r\3\r\3\r\3\r\3\16\3\16\3\16\3\17\3\17\3\17\3\17\3\20\3"+
		"\20\3\20\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\21\3\22\3\22\3\22\3\22\3"+
		"\22\3\22\3\22\3\23\3\23\3\23\3\23\3\23\3\23\3\23\3\24\3\24\3\24\3\24\3"+
		"\24\3\25\3\25\3\25\3\25\3\25\3\25\3\26\3\26\3\26\3\26\3\26\3\26\3\26\3"+
		"\26\3\26\3\26\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3"+
		"\27\3\27\3\27\3\27\3\27\3\27\3\27\5\27\u00b4\n\27\3\30\3\30\3\30\3\30"+
		"\3\30\3\30\3\30\3\30\3\30\3\30\3\30\3\30\3\30\5\30\u00c3\n\30\3\31\3\31"+
		"\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\31"+
		"\3\31\5\31\u00d6\n\31\3\32\3\32\3\32\3\32\5\32\u00dc\n\32\3\33\3\33\3"+
		"\34\3\34\3\35\6\35\u00e3\n\35\r\35\16\35\u00e4\3\36\6\36\u00e8\n\36\r"+
		"\36\16\36\u00e9\3\36\3\36\6\36\u00ee\n\36\r\36\16\36\u00ef\3\37\3\37\3"+
		"\37\3\37\3\37\3\37\3\37\3\37\3\37\5\37\u00fb\n\37\3 \3 \3 \3 \3 \3 \3"+
		" \3 \3 \3 \3 \3 \5 \u0109\n \3 \3 \7 \u010d\n \f \16 \u0110\13 \3 \3 "+
		"\3 \3!\3!\7!\u0117\n!\f!\16!\u011a\13!\3\"\3\"\7\"\u011e\n\"\f\"\16\""+
		"\u0121\13\"\3#\3#\3#\7#\u0126\n#\f#\16#\u0129\13#\3$\6$\u012c\n$\r$\16"+
		"$\u012d\3$\3$\2\2%\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27\r"+
		"\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33"+
		"\65\34\67\359\36;\37= ?!A\"C#E$G%\3\2\b\4\2((~~\4\2,,\61\61\4\2--//\4"+
		"\2$$^^\6\2\62;C\\aac|\5\2\13\f\17\17\"\"\u0149\2\3\3\2\2\2\2\5\3\2\2\2"+
		"\2\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3"+
		"\2\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2"+
		"\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2"+
		"\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2"+
		"\2\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2"+
		"\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\3I\3\2\2\2\5N\3\2\2\2"+
		"\7P\3\2\2\2\tR\3\2\2\2\13U\3\2\2\2\rW\3\2\2\2\17Y\3\2\2\2\21[\3\2\2\2"+
		"\23]\3\2\2\2\25_\3\2\2\2\27c\3\2\2\2\31e\3\2\2\2\33l\3\2\2\2\35o\3\2\2"+
		"\2\37s\3\2\2\2!v\3\2\2\2#~\3\2\2\2%\u0085\3\2\2\2\'\u008c\3\2\2\2)\u0091"+
		"\3\2\2\2+\u0097\3\2\2\2-\u00b3\3\2\2\2/\u00c2\3\2\2\2\61\u00d5\3\2\2\2"+
		"\63\u00db\3\2\2\2\65\u00dd\3\2\2\2\67\u00df\3\2\2\29\u00e2\3\2\2\2;\u00e7"+
		"\3\2\2\2=\u00fa\3\2\2\2?\u00fc\3\2\2\2A\u0114\3\2\2\2C\u011b\3\2\2\2E"+
		"\u0122\3\2\2\2G\u012b\3\2\2\2IJ\7h\2\2JK\7t\2\2KL\7q\2\2LM\7o\2\2M\4\3"+
		"\2\2\2NO\7_\2\2O\6\3\2\2\2PQ\7+\2\2Q\b\3\2\2\2RS\7?\2\2ST\7@\2\2T\n\3"+
		"\2\2\2UV\7\60\2\2V\f\3\2\2\2WX\7.\2\2X\16\3\2\2\2YZ\7]\2\2Z\20\3\2\2\2"+
		"[\\\7<\2\2\\\22\3\2\2\2]^\7*\2\2^\24\3\2\2\2_`\7p\2\2`a\7q\2\2ab\7v\2"+
		"\2b\26\3\2\2\2cd\7=\2\2d\30\3\2\2\2ef\7y\2\2fg\7k\2\2gh\7v\2\2hi\7j\2"+
		"\2ij\7k\2\2jk\7p\2\2k\32\3\2\2\2lm\7c\2\2mn\7u\2\2n\34\3\2\2\2op\7c\2"+
		"\2pq\7p\2\2qr\7f\2\2r\36\3\2\2\2st\7<\2\2tu\7?\2\2u \3\2\2\2vw\7d\2\2"+
		"wx\7g\2\2xy\7v\2\2yz\7y\2\2z{\7g\2\2{|\7g\2\2|}\7p\2\2}\"\3\2\2\2~\177"+
		"\7C\2\2\177\u0080\7u\2\2\u0080\u0081\7u\2\2\u0081\u0082\7k\2\2\u0082\u0083"+
		"\7i\2\2\u0083\u0084\7p\2\2\u0084$\3\2\2\2\u0085\u0086\7F\2\2\u0086\u0087"+
		"\7g\2\2\u0087\u0088\7h\2\2\u0088\u0089\7k\2\2\u0089\u008a\7p\2\2\u008a"+
		"\u008b\7g\2\2\u008b&\3\2\2\2\u008c\u008d\7H\2\2\u008d\u008e\7t\2\2\u008e"+
		"\u008f\7q\2\2\u008f\u0090\7o\2\2\u0090(\3\2\2\2\u0091\u0092\7Y\2\2\u0092"+
		"\u0093\7j\2\2\u0093\u0094\7g\2\2\u0094\u0095\7t\2\2\u0095\u0096\7g\2\2"+
		"\u0096*\3\2\2\2\u0097\u0098\7E\2\2\u0098\u0099\7q\2\2\u0099\u009a\7p\2"+
		"\2\u009a\u009b\7u\2\2\u009b\u009c\7w\2\2\u009c\u009d\7o\2\2\u009d\u009e"+
		"\7k\2\2\u009e\u009f\7p\2\2\u009f\u00a0\7i\2\2\u00a0,\3\2\2\2\u00a1\u00a2"+
		"\7u\2\2\u00a2\u00a3\7v\2\2\u00a3\u00a4\7t\2\2\u00a4\u00a5\7k\2\2\u00a5"+
		"\u00a6\7p\2\2\u00a6\u00b4\7i\2\2\u00a7\u00a8\7k\2\2\u00a8\u00a9\7p\2\2"+
		"\u00a9\u00b4\7v\2\2\u00aa\u00ab\7h\2\2\u00ab\u00ac\7n\2\2\u00ac\u00ad"+
		"\7q\2\2\u00ad\u00ae\7c\2\2\u00ae\u00b4\7v\2\2\u00af\u00b0\7d\2\2\u00b0"+
		"\u00b1\7q\2\2\u00b1\u00b2\7q\2\2\u00b2\u00b4\7n\2\2\u00b3\u00a1\3\2\2"+
		"\2\u00b3\u00a7\3\2\2\2\u00b3\u00aa\3\2\2\2\u00b3\u00af\3\2\2\2\u00b4."+
		"\3\2\2\2\u00b5\u00b6\7g\2\2\u00b6\u00b7\7c\2\2\u00b7\u00b8\7e\2\2\u00b8"+
		"\u00c3\7j\2\2\u00b9\u00ba\7n\2\2\u00ba\u00bb\7c\2\2\u00bb\u00bc\7u\2\2"+
		"\u00bc\u00c3\7v\2\2\u00bd\u00be\7h\2\2\u00be\u00bf\7k\2\2\u00bf\u00c0"+
		"\7t\2\2\u00c0\u00c1\7u\2\2\u00c1\u00c3\7v\2\2\u00c2\u00b5\3\2\2\2\u00c2"+
		"\u00b9\3\2\2\2\u00c2\u00bd\3\2\2\2\u00c3\60\3\2\2\2\u00c4\u00c5\7C\2\2"+
		"\u00c5\u00c6\7X\2\2\u00c6\u00d6\7I\2\2\u00c7\u00c8\7U\2\2\u00c8\u00c9"+
		"\7W\2\2\u00c9\u00d6\7O\2\2\u00ca\u00cb\7O\2\2\u00cb\u00cc\7C\2\2\u00cc"+
		"\u00d6\7Z\2\2\u00cd\u00ce\7O\2\2\u00ce\u00cf\7K\2\2\u00cf\u00d6\7P\2\2"+
		"\u00d0\u00d1\7E\2\2\u00d1\u00d2\7Q\2\2\u00d2\u00d3\7W\2\2\u00d3\u00d4"+
		"\7P\2\2\u00d4\u00d6\7V\2\2\u00d5\u00c4\3\2\2\2\u00d5\u00c7\3\2\2\2\u00d5"+
		"\u00ca\3\2\2\2\u00d5\u00cd\3\2\2\2\u00d5\u00d0\3\2\2\2\u00d6\62\3\2\2"+
		"\2\u00d7\u00dc\4>@\2\u00d8\u00d9\7#\2\2\u00d9\u00dc\7?\2\2\u00da\u00dc"+
		"\t\2\2\2\u00db\u00d7\3\2\2\2\u00db\u00d8\3\2\2\2\u00db\u00da\3\2\2\2\u00dc"+
		"\64\3\2\2\2\u00dd\u00de\t\3\2\2\u00de\66\3\2\2\2\u00df\u00e0\t\4\2\2\u00e0"+
		"8\3\2\2\2\u00e1\u00e3\4\62;\2\u00e2\u00e1\3\2\2\2\u00e3\u00e4\3\2\2\2"+
		"\u00e4\u00e2\3\2\2\2\u00e4\u00e5\3\2\2\2\u00e5:\3\2\2\2\u00e6\u00e8\4"+
		"\62;\2\u00e7\u00e6\3\2\2\2\u00e8\u00e9\3\2\2\2\u00e9\u00e7\3\2\2\2\u00e9"+
		"\u00ea\3\2\2\2\u00ea\u00eb\3\2\2\2\u00eb\u00ed\7\60\2\2\u00ec\u00ee\4"+
		"\62;\2\u00ed\u00ec\3\2\2\2\u00ee\u00ef\3\2\2\2\u00ef\u00ed\3\2\2\2\u00ef"+
		"\u00f0\3\2\2\2\u00f0<\3\2\2\2\u00f1\u00f2\7h\2\2\u00f2\u00f3\7c\2\2\u00f3"+
		"\u00f4\7n\2\2\u00f4\u00f5\7u\2\2\u00f5\u00fb\7g\2\2\u00f6\u00f7\7v\2\2"+
		"\u00f7\u00f8\7t\2\2\u00f8\u00f9\7w\2\2\u00f9\u00fb\7g\2\2\u00fa\u00f1"+
		"\3\2\2\2\u00fa\u00f6\3\2\2\2\u00fb>\3\2\2\2\u00fc\u010e\7$\2\2\u00fd\u0108"+
		"\7^\2\2\u00fe\u00ff\7t\2\2\u00ff\u0109\b \2\2\u0100\u0101\7p\2\2\u0101"+
		"\u0109\b \3\2\u0102\u0103\7v\2\2\u0103\u0109\b \4\2\u0104\u0105\7^\2\2"+
		"\u0105\u0109\b \5\2\u0106\u0107\7$\2\2\u0107\u0109\b \6\2\u0108\u00fe"+
		"\3\2\2\2\u0108\u0100\3\2\2\2\u0108\u0102\3\2\2\2\u0108\u0104\3\2\2\2\u0108"+
		"\u0106\3\2\2\2\u0109\u010d\3\2\2\2\u010a\u010b\n\5\2\2\u010b\u010d\b "+
		"\7\2\u010c\u00fd\3\2\2\2\u010c\u010a\3\2\2\2\u010d\u0110\3\2\2\2\u010e"+
		"\u010c\3\2\2\2\u010e\u010f\3\2\2\2\u010f\u0111\3\2\2\2\u0110\u010e\3\2"+
		"\2\2\u0111\u0112\7$\2\2\u0112\u0113\b \b\2\u0113@\3\2\2\2\u0114\u0118"+
		"\4C\\\2\u0115\u0117\t\6\2\2\u0116\u0115\3\2\2\2\u0117\u011a\3\2\2\2\u0118"+
		"\u0116\3\2\2\2\u0118\u0119\3\2\2\2\u0119B\3\2\2\2\u011a\u0118\3\2\2\2"+
		"\u011b\u011f\4c|\2\u011c\u011e\t\6\2\2\u011d\u011c\3\2\2\2\u011e\u0121"+
		"\3\2\2\2\u011f\u011d\3\2\2\2\u011f\u0120\3\2\2\2\u0120D\3\2\2\2\u0121"+
		"\u011f\3\2\2\2\u0122\u0123\7&\2\2\u0123\u0127\4c|\2\u0124\u0126\t\6\2"+
		"\2\u0125\u0124\3\2\2\2\u0126\u0129\3\2\2\2\u0127\u0125\3\2\2\2\u0127\u0128"+
		"\3\2\2\2\u0128F\3\2\2\2\u0129\u0127\3\2\2\2\u012a\u012c\t\7\2\2\u012b"+
		"\u012a\3\2\2\2\u012c\u012d\3\2\2\2\u012d\u012b\3\2\2\2\u012d\u012e\3\2"+
		"\2\2\u012e\u012f\3\2\2\2\u012f\u0130\b$\t\2\u0130H\3\2\2\2\25\2\u00b3"+
		"\u00c2\u00d5\u00db\u00e4\u00e9\u00ef\u00fa\u0108\u010c\u010e\u0116\u0118"+
		"\u011d\u011f\u0125\u0127\u012d\n\3 \2\3 \3\3 \4\3 \5\3 \6\3 \7\3 \b\b"+
		"\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}