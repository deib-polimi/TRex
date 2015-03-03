//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Gianpaolo Cugola, Daniele Rogora
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

package polimi.trex.examples;

import java.util.List;
import java.util.ArrayList;
import java.util.Date;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

import polimi.trex.common.Attribute;
import polimi.trex.common.Consts.EngineType;
import polimi.trex.communication.PacketListener;
import polimi.trex.communication.TransportManager;
import polimi.trex.packets.PubPkt;
import polimi.trex.packets.RulePkt;
import polimi.trex.packets.SubPkt;
import polimi.trex.packets.TRexPkt;
import polimi.trex.ruleparser.TRexRuleParser;

/**
 * @authors Gianpaolo Cugola, Daniele Rogora
 * 
 * A very basic, command line oriented, client for TRex.
 */
public class CommandLineClient implements PacketListener {
	static String teslaRule;
	static String readFile(String path, Charset encoding) 
			  throws IOException 
			{
			  byte[] encoded = Files.readAllBytes(Paths.get(path));
			  return encoding.decode(ByteBuffer.wrap(encoded)).toString();
			}
	
	private TransportManager tManager = new TransportManager(true);
    public static void main(String[] args) throws IOException {
   	String serverHost = null;
	int serverPort = -1;
	List<Integer> subTypes = null;
	int pubType = -1;
	List<String> keys=null, values=null;
	CommandLineClient client;
	int i = 0;
	Boolean sendRule = false;
	try {
	    subTypes = new ArrayList<Integer>();
	    pubType = -1;
	    keys = new ArrayList<String>();
	    values = new ArrayList<String>();
	    if(args.length<2) printUsageAndExit();
	    serverHost = args[i++];
	    serverPort = Integer.parseInt(args[i++]);
	    while(i<args.length) {
		if(i<args.length && args[i].equals("-pub")) {
		    i++;
		    pubType = Integer.parseInt(args[i++]);
		    while(i<args.length && !args[i].equals("-sub")) {
		    //System.out.println("Adding key " + args[i]);
			keys.add(args[i++]);
			//System.out.println("Adding value " + args[i]);
			values.add(args[i++]);
		    }
		}
		if(i<args.length && args[i].equals("-sub")) {
		    i++;
		    while(i<args.length && !args[i].equals("-sub")) {
			subTypes.add(Integer.parseInt(args[i++]));
		    }
		}
		if(i<args.length && args[i].equals("-rule")) {
		    i++;
		    sendRule = true;
		    teslaRule = readFile(args[i], Charset.defaultCharset());
		    i++;
		}
	    }
	} catch(NumberFormatException e) {
	    System.out.println("Error at parameter "+i);
	    printUsageAndExit();
	}
	try {
	    client = new CommandLineClient(serverHost, serverPort);
	    if(subTypes.size()>0) {
		client.tManager.addPacketListener(client);
		client.tManager.start();
		client.subscribe(subTypes);
	    }
	    if (sendRule) client.sendRule();
	    if(pubType!=-1) client.publish(pubType, keys, values);
	} catch(IOException e) { e.printStackTrace(); }
    }

    private static void printUsageAndExit() {
	System.out.println("Usage: java -jar TRexClient-JavaEx.jar "+
			   "<server_host> <server_port> "+
			   "[-rule path/to/file]"+
			   "[-sub <evt_type_1> ... <evt_type_n>]"+
			   "[-pub <evt_type> [<key_1> <val_1> ... <key_n> <val_n>]]");
	System.exit(-1);
    }

    public CommandLineClient(String serverHost, int serverPort) throws IOException {
    	tManager.connect(serverHost, serverPort);
    }
 
    public void sendRule() {
    	RulePkt rule = TRexRuleParser.parse(teslaRule, 2000);
    	try {
			tManager.sendRule(rule, EngineType.CPU);
		} catch (IOException e) { 
			e.printStackTrace();
		}
    }
    
    public void subscribe(List<Integer> subTypes) {
	for(int subType : subTypes) {
		SubPkt sub = new SubPkt(subType);
	    try {
		    tManager.send(sub);	
		 } catch (IOException e) { e.printStackTrace(); }
	    }
    }
    
    public void publish(int pubType, List<String> keys, List<String> values) {
	PubPkt pub;
	boolean boolVal;
	int intVal;
	float floatVal;

	pub = new PubPkt(pubType);
	for(int i=0; i<keys.size(); i++) {
	    if(values.get(i).equals("true")) {
		boolVal = true;
		pub.addAttribute(new Attribute(keys.get(i), boolVal)); // add a bool attr
	    } else if(values.get(i).equals("false")) {
		boolVal = false;
		pub.addAttribute(new Attribute(keys.get(i), boolVal)); // add a bool attr
	    } else {
		try {
		    intVal = Integer.parseInt(values.get(i));
		    pub.addAttribute(new Attribute(keys.get(i), intVal)); // add an int attr
		} catch(NumberFormatException e1) {
		    try {
			floatVal = Float.parseFloat(values.get(i));
			pub.addAttribute(new Attribute(keys.get(i), floatVal)); // add a float attr
		    } catch(NumberFormatException e2) {
			pub.addAttribute(new Attribute(keys.get(i), values.get(i))); // add a String attr
		    }
		}
	    }
	}
	try {
	    tManager.send(pub);
	} catch (IOException e) { e.printStackTrace(); }	
    }

    @Override
    public void notifyPktReceived(TRexPkt pkt) {
	if(! (pkt instanceof PubPkt)) {
	    System.out.println("Ingnoring wrong packet: "+pkt);
	    return;
	}
	PubPkt pub = (PubPkt) pkt;
	System.out.print("PubPacket received: {");
	System.out.print(pub.getEventType());
	for(Attribute att : pub.getAttributes()) {
	    System.out.print(" <"+att.getName());
	    switch(att.getValType()) {
	    case BOOL: System.out.print(" : bool = "+att.getBoolVal()+">"); break;
	    case INT: System.out.print(" : int = "+att.getIntVal()+">"); break;
	    case FLOAT: System.out.print(" : float = "+att.getFloatVal()+">"); break;
	    case STRING: System.out.print(" : string = "+att.getStringVal()+">"); break;
	    }
	}
	System.out.print("}@");
	System.out.println(new Date(pub.getTimeStamp()).toLocaleString());
    }
    @Override
    public void notifyConnectionError() {
	System.out.println("Connection error. Exiting.");
	System.exit(-1);
    }
}
    
