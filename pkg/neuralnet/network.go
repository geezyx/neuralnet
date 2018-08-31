package neuralnet

import (
	"fmt"
	"math/rand"
)

type Network struct {
	Inputs  []*Synapse
	Outputs []*Synapse
	Layers  []Layer
}

type Layer struct {
	Neurons []*Neuron
}

func NewNetwork(inputs, outputs []*float64, size int) *Network {
	network := &Network{
		Inputs:  []*Synapse{},
		Outputs: []*Synapse{},
		Layers:  []Layer{},
	}

	network.addInputLayer(inputs)
	for i := 0; i < size; i++ {
		network.addHiddenLayer(2)
	}
	network.addOutputLayer(outputs)

	network.connectLayers()
	return network
}

func (n *Network) addInputLayer(inputs []*float64) {
	l := Layer{
		Neurons: []*Neuron{},
	}

	b := NewBias(1)
	l.Neurons = append(l.Neurons, b)

	weight := float64(1.0)

	for _, input := range inputs {
		s := &Synapse{Value: input, Weight: &weight}
		n.Inputs = append(n.Inputs, s)
		n := NewInput(s)
		l.Neurons = append(l.Neurons, n)
	}

	n.Layers = append(n.Layers, l)
}

func (n *Network) addHiddenLayer(size int) {
	l := Layer{
		Neurons: []*Neuron{},
	}

	b := NewBias(1)
	l.Neurons = append(l.Neurons, b)

	for i := 0; i < size; i++ {
		n := NewSigmoid()
		l.Neurons = append(l.Neurons, n)
	}

	n.Layers = append(n.Layers, l)
}

func (n *Network) addOutputLayer(outputs []*float64) {
	l := Layer{
		Neurons: []*Neuron{},
	}

	weight := float64(1.0)

	for _, output := range outputs {
		s := &Synapse{Value: output, Weight: &weight}
		n.Outputs = append(n.Outputs, s)
		n := NewOutput(s)
		l.Neurons = append(l.Neurons, n)
	}
}

func (n *Network) connectLayers() {
	for i := 0; i < len(n.Layers)-1; i++ {
		currentLayer := n.Layers[i]
		nextLayer := n.Layers[i+1]
		for _, neuron := range currentLayer.Neurons {
			for _, target := range nextLayer.Neurons {
				s := &Synapse{}
				neuron.Outputs = append(neuron.Outputs, s)
				target.Inputs = append(target.Inputs, s)
				randomFloat := rand.Float64()
				s.Weight = &randomFloat
			}
		}
	}
}

func (n Network) String() string {
	var output string
	for i, l := range n.Layers {
		output += fmt.Sprintf("Layer %d: ", i)
		for _, n := range l.Neurons {
			for _, s := range n.Inputs {
				output += fmt.Sprintf("( %.2f | %.2f )", *s.Value, *s.Weight)
			}
		}
		output += "\n"
	}
	return output
}

func (n *Network) Process() {
	for _, l := range n.Layers {
		for _, n := range l.Neurons {
			n.Function(n.Inputs, n.Outputs)
		}
	}
}
