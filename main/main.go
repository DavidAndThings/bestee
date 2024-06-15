package main

import (
	"bestee/logic"
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

func (bestee *Bestee) GetEntityBank() *logic.EntityBank {

	executablePath := util.GetExcutableDir()
	entities := logic.NewEmptyEntityBank()

	for _, filePath := range bestee.config["entity_files"].([]interface{}) {
		entities.AddFromJsonFile(executablePath + filePath.(string))
	}

	return entities

}

func main() {

	bestee := BesteeFromConfigFile()
	fmt.Println(bestee.GetEntityBank())
}
