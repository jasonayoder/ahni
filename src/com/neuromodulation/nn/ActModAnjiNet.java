package com.neuromodulation.nn;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import com.anji.nn.AnjiNet;
import com.anji.nn.CacheNeuronConnection;
import com.anji.nn.Neuron;
import com.anji.nn.NeuronConnection;

public class ActModAnjiNet extends AnjiNet {
	// UROC ZachSierp: Contains all Source code from AnjiNet
	//
	// Removed functions that can be handled at parent level.
	//
	/**
	 * base XML tag
	 */
	public final static String XML_TAG = "network";
	
	private ArrayList<Neuron> allNeurons;
	
	//UROC ZachSierp: Add array list for all connections
	private ArrayList<NeuronConnection> allConns;

	private List<CacheNeuronConnection> recurrentConns;

	/**
	 * @param allNeurons
	 * @param inputNeurons
	 * @param outputNeurons
	 * @param recurrentConns
	 * @param aName
	 */
	//UROC ZachSierp: Added @param allConns. Changed Constructor to ActModAnjiNet
	public ActModAnjiNet(Collection<Neuron> allNeurons, List<Neuron> inputNeurons, 
			List<Neuron> outputNeurons, List<CacheNeuronConnection> recurrentConns, 
				Collection<NeuronConnection> allConns, String aName) {
		
		super(allNeurons, inputNeurons, outputNeurons, recurrentConns, aName);
		allConns = new ArrayList<NeuronConnection>(allConns);
		
	}

	/**
	 * @param someNeurons all neurons
	 * @param someInNeurons input neurons (also included in someNeurons)
	 * @param someOutNeurons output neurons (also included in someNeurons)
	 * @param someRecurrentConns recurrent connections
	 * @param aName
	 */
	
	//UROC ZachSierp: returns collection of all connections
	public Collection<NeuronConnection> getConns() {
		return allConns;
	}
	
	//UROC ZachSierp: return specific connection in allConns ArrayList
	public NeuronConnection getConn(int idx) {
		return allConns.get(idx);
	}

	/**
	 * indicates a time step has passed
	 */
	public void step() {
		try {
			// populate cache connections with values from previous step
			// We don't use the Collections iterator functionality because it's slower for small collections.
			for (int i = 0 ; i < recurrentConns.size(); i++) {
				recurrentConns.get(i).step();
				//System.out.println("recurrent! " + name);
			}
			for (int i = 0 ; i < allNeurons.size(); i++) {
				allNeurons.get(i).step();
			}
		}
		catch (StackOverflowError e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
			System.err.println("step() AnjiNet:\n" + toXml());
		}
	}

	/**
	 * make sure all neurons have been activated for the current cycle; this is to catch neurons with no forward outputs
	 */
	public void fullyActivate() {
		try {
			// We don't use the Collections iterator functionality because it's slower for small collections.
			for (int i = 0 ; i < allNeurons.size(); i++) {
				allNeurons.get(i).getValue();
			}
		}
		catch (StackOverflowError e) {
			System.err.println(e.getMessage());
			e.printStackTrace();
			System.err.println("fA() AnjiNet:\n" + toXml());
		}
	}

	/**
	 * clear all memory in network, including neurons and recurrent connections
	 */
	//UROC ZachSierp: added functionality to clear allConns list of values
	public void reset() {
		// We don't use the Collections iterator functionality because it's slower for small collections.
		for (int i = 0 ; i < allNeurons.size(); i++) {
			allNeurons.get(i).reset();
		}
		for (int i = 0 ; i < recurrentConns.size(); i++) {
			recurrentConns.get(i).reset();
		}
		//UROC ZachSierp: set all values in allConns to null
		allConns.clear();
	}
	
	// UROC ZachSierp: return true if network contains any modulating connections, else false
	// getTypeFunction() is a placeholder for code layout. 
	public boolean isModulating() {
		boolean res = false;
		for (NeuronConnection c : allConns) {
			if (c instanceof ActModNeuronConnection) {
				res = true;
				break;
			}
		}
		return res;
	}
}


