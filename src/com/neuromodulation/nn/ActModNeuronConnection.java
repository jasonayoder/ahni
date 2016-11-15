package com.neuromodulation.nn;

import com.anji.nn.Neuron;
import com.anji.nn.NeuronConnection;

public class ActModNeuronConnection extends NeuronConnection {

	public ActModNeuronConnection(Neuron anIncoming) {
		super(anIncoming);
	}

	
	public ActModNeuronConnection(Neuron anIncoming, double aWeight) {
		super(anIncoming, aWeight);
	}

}
