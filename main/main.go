package main

import (
	"bestee/core"
	"bestee/util"
	"fmt"
)

type Bestee struct {
	config map[string]interface{}
}

func BesteeFromConfigFile() *Bestee {
	configPath := util.GetExcutableDir() + "/config.json"
	return &Bestee{config: util.ReadJsonIntoMap(configPath)}
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

func main() {

	bestee := BesteeFromConfigFile()
	fmt.Println(bestee.GetEntityBank())
}
