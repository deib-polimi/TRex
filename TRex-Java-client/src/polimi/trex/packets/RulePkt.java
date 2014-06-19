//
// This file is part of T-Rex, a Complex Event Processing Middleware.
// See http://home.dei.polimi.it/margara
//
// Authors: Alessandro Margara
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

package polimi.trex.packets;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import polimi.trex.common.TAggregate;
import polimi.trex.common.ComplexParameter;
import polimi.trex.common.EventPredicate;
import polimi.trex.common.EventTemplate;
import polimi.trex.common.Negation;
import polimi.trex.common.Parameter;


/**
 * Defines a RulePkt, used to send a rule to the T-Rex engine
 */
public class RulePkt implements TRexPkt {
	private Map<Integer, EventPredicate> predicates;
	private Map<Integer, ComplexParameter> parameters;
	private Map<Integer, Negation> negations;
	private Map<Integer, TAggregate> aggregates;
	Set<Integer> consuming;
	EventTemplate eventTemplate;

	public RulePkt(EventTemplate eventTemplate) {
		this.eventTemplate = eventTemplate;
		predicates = new HashMap<Integer, EventPredicate>();
		parameters = new HashMap<Integer, ComplexParameter>();
		negations = new HashMap<Integer, Negation>();
		aggregates = new HashMap<Integer, TAggregate>();
		consuming = new HashSet<Integer>();
	}
	
	public int getPredicatesNum() {
		return predicates.size();
	}

	public EventPredicate getPredicates(int index) {
		return predicates.get(index);
	}

	public void addPredicate(EventPredicate predicate) {
		predicates.put(predicates.size(), predicate);
	}

	public int getParametersNum() {
		return parameters.size();
	}

	public ComplexParameter getParameter(int index) {
		return parameters.get(index);
	}

	public void addParameter(ComplexParameter parameter) {
		parameters.put(parameters.size(), parameter);
	}
	
	public int getNegationsNum() {
		return negations.size();
	}

	public Negation getNegation(int index) {
		return negations.get(index);
	}

	public void addNegation(Negation negation) {
		negations.put(negations.size(), negation);
	}
	
	public int getAggregatesNum() {
		return aggregates.size();
	}

	public TAggregate getAggregate(int index) {
		return aggregates.get(index);
	}

	public void addAggregate(TAggregate aggregate) {
		aggregates.put(aggregates.size(), aggregate);
	}
	
	public Collection<Integer> getConsuming() {
		return consuming;
	}
	
	public void addConsuming(int consumingIndex) {
		consuming.add(consumingIndex);
	}
	
	public EventTemplate getEventTemplate() {
		return eventTemplate;
	}
	
	public void setEventTemplate(EventTemplate et) {
		this.eventTemplate = et;
	}
	
	public Map<Integer, EventPredicate> getPredicates() {
		return predicates;
	}

	public Map<Integer, ComplexParameter> getParameters() {
		return parameters;
	}

	public Map<Integer, Negation> getNegations() {
		return negations;
	}

	public Map<Integer, TAggregate> getAggregates() {
		return aggregates;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (!(obj instanceof RulePkt)) return false;
		RulePkt other = (RulePkt) obj;
		if (predicates.size() != other.predicates.size()) return false;
		for (Integer key : predicates.keySet()) {
			if (! other.predicates.containsKey(key)) return false;
			if (! other.predicates.get(key).equals(predicates.get(key))) return false;
		}
		if (aggregates.size() != other.aggregates.size()) return false;
		for (Integer key : aggregates.keySet()) {
			if (! other.aggregates.containsKey(key)) return false;
			if (! other.aggregates.get(key).equals(aggregates.get(key))) return false;
		}
		if (negations.size() != other.negations.size()) return false;
		for (Integer key : negations.keySet()) {
			if (! other.negations.containsKey(key)) return false;
			if (! other.negations.get(key).equals(negations.get(key))) return false;
		}
		if (parameters.size() != other.parameters.size()) return false;
		for (Integer key : parameters.keySet()) {
			if (! other.parameters.containsKey(key)) return false;
			if (! other.parameters.get(key).equals(parameters.get(key))) return false;
		}
		if (consuming.size() != other.consuming.size()) return false;
		if (! consuming.containsAll(other.consuming)) return false;
		if (! eventTemplate.equals(other.eventTemplate)) return false;
		return true;
	}
}
