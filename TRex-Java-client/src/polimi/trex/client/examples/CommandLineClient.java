package polimi.trex.client.examples;

import java.util.HashSet;
import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;

import polimi.trex.client.communication.PacketListener;
import polimi.trex.client.communication.TransportManager;
import polimi.trex.client.packets.Packet;
import polimi.trex.client.packets.PubPacket;
import polimi.trex.client.packets.RulePacket;
import polimi.trex.client.packets.SubPacket;
import polimi.trex.client.ruleparser.TRexRuleParser;
import polimi.trex.common.Attribute;
import polimi.trex.common.Constraint;
import polimi.trex.common.Consts.CompKind;
import polimi.trex.common.Consts.ConstraintOp;
import polimi.trex.common.Consts.EngineType;
import polimi.trex.common.Consts.StateType;
import polimi.trex.common.Consts.ValType;
import polimi.trex.common.ComplexParameter;
import polimi.trex.common.EventPredicate;
import polimi.trex.common.EventTemplate;
import polimi.trex.common.EventTemplateAttr;
import polimi.trex.common.EventTemplateStaticAttr;
import polimi.trex.common.OpTree;
import polimi.trex.common.RulePktValueReference;
import polimi.trex.common.StaticValueReference;
import polimi.trex.packets.RulePkt;

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
    	RulePacket rule = TRexRuleParser.parse(teslaRule, 2000);
    	try {
			tManager.sendRule(rule, EngineType.CPU);
		} catch (IOException e) {
			e.printStackTrace();
		}
    }
    
    public void subscribe(List<Integer> subTypes) {
	for(int subType : subTypes) {
	    SubPacket sub = new SubPacket(subType);
	    try {
		tManager.send(sub);
	    } catch (IOException e) { e.printStackTrace(); }
	}
    }
    
    public void publish(int pubType, List<String> keys, List<String> values) {
	PubPacket pub;
	boolean boolVal;
	int intVal;
	float floatVal;

	pub = new PubPacket(pubType);
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
    public void notifyPktReceived(Packet pkt) {
	if(! (pkt instanceof PubPacket)) {
	    System.out.println("Ingnoring wrong packet: "+pkt);
	    return;
	}
	PubPacket pub = (PubPacket) pkt;
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
	System.out.println("}");
    }
    @Override
    public void notifyConnectionError() {
	System.out.println("Connection error. Exiting.");
	System.exit(-1);
    }
}
    
