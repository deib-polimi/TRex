package polimi.trex.client.packets;

import polimi.trex.common.EventTemplate;
import polimi.trex.packets.RulePkt;

public class RulePacket extends RulePkt implements Packet {

	public RulePacket(EventTemplate eventTemplate) {
		super(eventTemplate);
	}
	
	public RulePacket(RulePkt trexRulePkt) {
		this(trexRulePkt.getEventTemplate());
	}
}