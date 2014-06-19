/**
 * @author Daniele Rogora
 *
 */
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

package polimi.trex.client.ruleparser;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.antlr.v4.runtime.ANTLRInputStream;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.ParseTreeVisitor;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

import polimi.trex.client.packets.RulePacket;
import polimi.trex.common.EventTemplate;
import polimi.trex.packets.RulePkt;



public class TRexRuleParser {
	//public static int index=1000;
	
	/**
	 * Return a complete RulePacket describing the definition expressed in the given string.
	 * Event attributes, even for different event types, with the same name must have the same ValType (INT, FLOAT, BOOL or STRING)
	 * @param rule a String expressing the definition of the rule in TESLA
	 * @return a RulePacket
	 */
  public static RulePacket parse(String rule, int index) {	  
	  ANTLRInputStream input = new ANTLRInputStream(rule);
	  TESLALexer lexer = new TESLALexer(input);
	  CommonTokenStream tokens = new CommonTokenStream(lexer);
	  TESLAParser parser = new TESLAParser(tokens);
      ParseTree t = parser.trex_rule();
      ParseTreeWalker walker = new ParseTreeWalker(); // create standard walker
      
      List<Token> ttt = tokens.getTokens();
      Map<String, Integer> predicates = new HashMap<String, Integer>();
	  System.out.println("### NAMES -> IDS mapping ###\n");
	  for (Token tok : ttt) {
		  if (tok.getType()== TESLALexer.PRED_NAME) {
			  if (!predicates.containsKey(tok.getText())) {
				  predicates.put(tok.getText(), index);
				  index++;
				  System.out.println(tok.getText()+ " -> " + predicates.get(tok.getText()));
			  }
		  }
	  }
	  System.out.println("\n### END ###");
	  TESLABaseListener extractor = new TESLABaseListener(predicates);
	  walker.walk(extractor, t); // initiate walk of tree with listener
	  RulePacket res = extractor.getRule();
	  return res;
  }
}
