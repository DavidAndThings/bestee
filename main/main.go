package main

import (
	"bestee/core"
	"bufio"
	"fmt"
	"os"
	"strings"
)

type Bestee struct {
	keyboardInput chan string
}

func NewBestee() *Bestee {
	return &Bestee{
		keyboardInput: make(chan string),
	}
}

func (bestee *Bestee) GetMachine() *core.Machine {

	return core.NewMachineWithLogicBlocks(
		core.NewBinaryExchangeBlock(),
	)

}

func (bestee *Bestee) Run() {

	go bestee.machineLoop()
	bestee.keyboardInputLoop()

}

func (bestee *Bestee) machineLoop() {

	machine := bestee.GetMachine()

	for {
		select {
		case keyboardInput := <-bestee.keyboardInput:

			machine.AddToSignalQueue(core.BuildPlainText(keyboardInput))
			machine.RunEpoch()

		}
	}
}

func (bestee *Bestee) keyboardInputLoop() {

	for {

		fmt.Print("> ")
		reader := bufio.NewReader(os.Stdin)
		text, _ := reader.ReadString('\n')
		trimmedText := strings.Trim(text, "\n")

		bestee.keyboardInput <- trimmedText

	}

}

func main() {

	bestee := NewBestee()
	bestee.Run()

}
