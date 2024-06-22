package main

import (
	"bestee/core"
	"bestee/util"
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/google/uuid"
)

type Bestee struct {
	config        map[string]interface{}
	keyboardInput chan string
}

func BesteeFromConfigFile() *Bestee {
	configPath := util.GetExcutableDir() + "/config.json"
	return &Bestee{
		config:        util.ReadJsonIntoMap(configPath),
		keyboardInput: make(chan string),
	}
}

func (bestee *Bestee) GetMachine() *core.Machine {
	return core.NewMachineWithLogicBlocks(bestee.GetEntityBank())
}

func (bestee *Bestee) GetEntityBank() *core.EntityBank {

	executablePath := util.GetExcutableDir()
	entities := core.NewEmptyEntityBank()

	for _, filePath := range bestee.config["entity_files"].([]interface{}) {
		entities.AddFromJsonFile(executablePath + filePath.(string))
	}

	return entities

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

			machine.AddToSignalQueue(core.Expression{
				Header: core.ENTITY_SPECIFY,
				Data: map[string]interface{}{
					"_id":        uuid.New().String(),
					"first_name": keyboardInput,
				},
			})

			machine.RunEpoch()

		}
	}
}

func (bestee *Bestee) keyboardInputLoop() {

	for {

		reader := bufio.NewReader(os.Stdin)
		fmt.Println("Me: ")
		text, _ := reader.ReadString('\n')
		trimmedText := strings.Trim(text, "\n")

		bestee.keyboardInput <- trimmedText

	}

}

func main() {

	bestee := BesteeFromConfigFile()
	bestee.Run()

}
