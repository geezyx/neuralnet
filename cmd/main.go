package main

import (
	"fmt"

	"github.com/geezyx/neuralnet/pkg/neuralnet"
)

func main() {
	output1 := float64(0.0)

	input1 := float64(5.0)
	input2 := float64(10.0)

	n := neuralnet.NewNetwork([]*float64{&input1, &input2}, []*float64{&output1}, 2)
	fmt.Println(n)
	n.Process()
	fmt.Println(n)
	fmt.Println(output1)
	fmt.Println(input1)
}
