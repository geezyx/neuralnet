package neuralnet

import "math"

// Neuron represents a generic neuron in a network
// with input and output synapses and an activation function
type Neuron struct {
	Inputs   []*Synapse
	Outputs  []*Synapse
	Function func([]*Synapse, []*Synapse)
}

// Synapse represents a logical connection between
// Neurons, and carries a weight
type Synapse struct {
	Value  *float64
	Weight *float64
}

// NewSigmoid creates a neuron with a sigmoid function
func NewSigmoid() *Neuron {
	return &Neuron{
		Inputs:  []*Synapse{},
		Outputs: []*Synapse{},
		Function: func(inputs, outputs []*Synapse) {
			var sum float64
			for _, s := range inputs {
				sum += (*s.Value * *s.Weight)
			}
			out := sigmoid(sum)
			for _, s := range outputs {
				s.Value = &out
			}
		},
	}
}

// NewBias creates a basic bias neuron which will
// generate a constant value to all output synapses
func NewBias(val float64) *Neuron {
	weight := float64(1.0)
	return &Neuron{
		Inputs: []*Synapse{
			{
				Value:  &val,
				Weight: &weight,
			},
		},
		Outputs: []*Synapse{},
		Function: func(inputs, outputs []*Synapse) {
			for _, s := range outputs {
				s.Value = inputs[0].Value
				s.Weight = inputs[0].Weight
			}
		},
	}
}

// NewInput creates a basic input neuron
func NewInput(input *Synapse) *Neuron {
	return &Neuron{
		Inputs:  []*Synapse{input},
		Outputs: []*Synapse{},
		Function: func(inputs, outputs []*Synapse) {
			for _, s := range outputs {
				s.Value = inputs[0].Value
			}
		},
	}
}

// NewOutput creates a basic output neuron
func NewOutput(output *Synapse) *Neuron {
	return &Neuron{
		Inputs:  []*Synapse{},
		Outputs: []*Synapse{output},
		Function: func(inputs, outputs []*Synapse) {
			var sum float64
			for _, s := range inputs {
				sum += (*s.Value * *s.Weight)
			}
			outputs[0].Value = &sum
		},
	}
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}
