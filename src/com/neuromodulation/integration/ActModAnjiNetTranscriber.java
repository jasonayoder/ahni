package com.neuromodulation.integration;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;

import org.apache.log4j.Logger;
import org.jgapcustomised.Chromosome;

import com.anji.integration.AnjiActivator;
import com.anji.integration.AnjiNetTranscriber;
import com.anji.integration.TranscriberException;
import com.anji.neat.ConnectionAllele;
import com.anji.neat.NeatChromosomeUtility;
import com.anji.neat.NeuronAllele;
import com.anji.neat.NeuronType;
import com.anji.nn.CacheNeuronConnection;
import com.anji.nn.Neuron;
import com.anji.nn.NeuronConnection;
import com.anji.nn.RecurrencyPolicy;
import com.anji.nn.activationfunction.ActivationFunctionFactory;
import com.anji.util.Properties;
import com.neuromodulation.nn.*;

public class ActModAnjiNetTranscriber extends AnjiNetTranscriber {
	
	//UROC ZachSierp/Akelikian: get recurrentCycles value from props file
	Properties props = new Properties();
	private int recurrentCycles = props.getIntProperty(RECURRENT_CYCLES_KEY, 1);
	//get recurrency policy using parent function
	private RecurrencyPolicy recurrencyPolicy = RecurrencyPolicy.BEST_GUESS;
	//get Logger using parent class
	private final static Logger logger = Logger.getLogger(AnjiNetTranscriber.class);
	
	public AnjiActivator transcribe(Chromosome genotype) throws TranscriberException {
		return new ActModAnjiActivator(newActModAnjiNet(genotype), recurrentCycles);
	}
	
	public ActModAnjiNet newActModAnjiNet(Chromosome genotype) throws TranscriberException {
		if (genotype.getAlleles().isEmpty()) throw new IllegalArgumentException("Genotype has no alleles...");

		Map<Long, Neuron> allNeurons = new HashMap<Long, Neuron>();

		// input neurons
		SortedMap<Long, NeuronAllele> inNeuronAlleles = NeatChromosomeUtility.getNeuronMap(genotype.getAlleles(), NeuronType.INPUT);
		if (inNeuronAlleles.isEmpty()) throw new IllegalArgumentException("An AnjiNet must have at least one input neuron.");
		List<Neuron> inNeurons = new ArrayList<Neuron>();
		for (NeuronAllele neuronAllele : inNeuronAlleles.values()) {
			Neuron n = new Neuron(ActivationFunctionFactory.getInstance().get(neuronAllele.getActivationType()), neuronAllele.getBias());
			n.setId(neuronAllele.getInnovationId().longValue());
			inNeurons.add(n);
			allNeurons.put(neuronAllele.getInnovationId(), n);
		}

		// output neurons
		SortedMap<Long, NeuronAllele> outNeuronAlleles = NeatChromosomeUtility.getNeuronMap(genotype.getAlleles(), NeuronType.OUTPUT);
		if (outNeuronAlleles.isEmpty()) throw new IllegalArgumentException("An AnjiNet must have at least one output neuron.");
		List<Neuron> outNeurons = new ArrayList<Neuron>();
		for (NeuronAllele neuronAllele : outNeuronAlleles.values()) {
			Neuron n = new Neuron(ActivationFunctionFactory.getInstance().get(neuronAllele.getActivationType()), neuronAllele.getBias());
			n.setId(neuronAllele.getInnovationId().longValue());
			outNeurons.add(n);
			allNeurons.put(neuronAllele.getInnovationId(), n);
		}

		// hidden neurons
		SortedMap<Long, NeuronAllele> hiddenNeuronAlleles = NeatChromosomeUtility.getNeuronMap(genotype.getAlleles(), NeuronType.HIDDEN);
		for (NeuronAllele neuronAllele : hiddenNeuronAlleles.values()) {
			Neuron n = new Neuron(ActivationFunctionFactory.valueOf(neuronAllele.getActivationType()), neuronAllele.getBias());
			n.setId(neuronAllele.getInnovationId().longValue());
			allNeurons.put(neuronAllele.getInnovationId(), n);
		}

		// connections
		//
		// Starting with output layer, gather connections and neurons to which it is immediately
		// connected, creating a logical layer of neurons. Assign each connection its input (source)
		// neuron, and each destination neuron its input connections. Recurrency is handled depending
		// on policy:
		//
		// RecurrencyPolicy.LAZY - all connections CacheNeuronConnection
		//
		// RecurrencyPolicy.DISALLOWED - no connections CacheNeuronConnection (assumes topology sans
		// loops)
		//
		// RecurrencyPolicy.BEST_GUESS - any connection where the source neuron is in the same or
		// later (i.e., nearer output layer) as the destination is a CacheNeuronConnection
		List<CacheNeuronConnection> recurrentConns = new ArrayList<CacheNeuronConnection>();
		List<ConnectionAllele> remainingConnAlleles = NeatChromosomeUtility.getConnectionList(genotype.getAlleles());
		Set<Long> currentNeuronInnovationIds = new HashSet<Long>(outNeuronAlleles.keySet());
		Set<Long> traversedNeuronInnovationIds = new HashSet<Long>(currentNeuronInnovationIds);
		Set<Long> nextNeuronInnovationIds = new HashSet<Long>();
		
		// UROC ZachSierp/Akelikian: Create list of all connections for new constructor
		List<NeuronConnection> allConns = new ArrayList<NeuronConnection>();
		
		while (!remainingConnAlleles.isEmpty() && !currentNeuronInnovationIds.isEmpty()) {
			nextNeuronInnovationIds.clear();
			Collection<ConnectionAllele> connAlleles = NeatChromosomeUtility.extractConnectionAllelesForDestNeurons(remainingConnAlleles, currentNeuronInnovationIds);
			for (ConnectionAllele connAllele : connAlleles) {
				Neuron src = allNeurons.get(connAllele.getSrcNeuronId());
				Neuron dest = allNeurons.get(connAllele.getDestNeuronId());
				if (src == null)
					throw new TranscriberException("connection with missing src neuron: " + connAllele.toString());
				if (dest == null)
					throw new TranscriberException("connection with missing dest neuron: " + connAllele.toString());
				
				//UROC ZachSierp/Akelikian: create new ActModNeuronConnection using source allele and add it to 
				//allConns list. No visibility on recurrencyPolicy
				// Create new connection using src neuron and weight
				ActModNeuronConnection actConn = new ActModNeuronConnection(src, connAllele.getWeight());
				allConns.add(actConn);
				
				// handle recurrency processing
				boolean cached = false;
				if (RecurrencyPolicy.LAZY.equals(recurrencyPolicy))
					cached = true;
				else if (RecurrencyPolicy.BEST_GUESS.equals(recurrencyPolicy)) {
					boolean maybeRecurrent = (traversedNeuronInnovationIds.contains(connAllele.getSrcNeuronId()));
					cached = maybeRecurrent || recurrencyPolicy.equals(RecurrencyPolicy.LAZY);
				}
				NeuronConnection conn = null;
				if (cached) {					
					conn = new CacheNeuronConnection(src, (float) connAllele.getWeight());
					recurrentConns.add((CacheNeuronConnection) conn);
				} else {
					conn = new NeuronConnection(src, (float) connAllele.getWeight());
				}

				conn.setId(connAllele.getInnovationId().longValue());
				dest.addIncomingConnection(conn);
				nextNeuronInnovationIds.add(connAllele.getSrcNeuronId());
			}
			traversedNeuronInnovationIds.addAll(nextNeuronInnovationIds);
			currentNeuronInnovationIds.clear();
			currentNeuronInnovationIds.addAll(nextNeuronInnovationIds);
			remainingConnAlleles.removeAll(connAlleles);
		}

		// make sure we traversed all connections and nodes; input neurons are automatically
		// considered "traversed" since they should be realized regardless of their connectivity to
		// the rest of the network
		
		//UROC ZachSierp/Akelikian: No visibility on logger
		if (!remainingConnAlleles.isEmpty()) {
			logger.warn("not all connection genes handled: " + genotype.toString() + (genotype.getMaterial().pruned ? "  " : "  not") + " pruned");
		}
		traversedNeuronInnovationIds.addAll(inNeuronAlleles.keySet());
		if (traversedNeuronInnovationIds.size() != allNeurons.size()) {
			logger.warn("did not traverse all neurons: " + genotype.toString() + (genotype.getMaterial().pruned ? "  " : "  not") + " pruned");
		}

		// build network

		Collection<Neuron> allNeuronsCol = allNeurons.values();
		String id = genotype.getId().toString();
		//UROC ZachSierp/Akelikian: Changed to ActMod constructor and added allConns 
		return new ActModAnjiNet(allNeuronsCol, inNeurons, outNeurons, recurrentConns, allConns, id);
	}

}
