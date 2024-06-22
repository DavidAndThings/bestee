package core

import (
	"bestee/util"
	"fmt"
	"slices"
	"strings"
)

var entityBankInstance *EntityBank

type EntityBank struct {
	entities []map[string]interface{}
}

func NewEntityBank() *EntityBank {

	exeDir := util.GetExcutableDir()
	config := util.ReadConfigJson()
	entities := make([]map[string]interface{}, 0)

	for _, filePath := range config["entity_files"].([]interface{}) {
		entities = append(entities, util.ReadJsonIntoArray(exeDir+filePath.(string))...)
	}

	return &EntityBank{entities: entities}

}

func GetEntityBankInstance() *EntityBank {

	if entityBankInstance == nil {
		entityBankInstance = NewEntityBank()
	}

	return entityBankInstance

}

func (bank *EntityBank) AddFromJsonFile(filePath string) {
	bank.entities = append(bank.entities, util.ReadJsonIntoArray(filePath)...)
}

func (bank *EntityBank) Process(machine *Machine) []Expression {

	newExps := make([]Expression, 0)

	for id, specification := range machine.findUntranslatedSpecifications() {

		entityMatches, searchErr := bank.findEntitiesWithEntitySpecify(specification)

		if searchErr == nil {
			for _, match := range entityMatches {
				newExps = append(newExps, BuildEntityTranslate(id, match["_id"].(string)))
			}
		}

	}

	return newExps

}

func (bank *EntityBank) findEntitiesWithEntitySpecify(exp Expression) ([]map[string]interface{}, error) {

	matches := make([]map[string]interface{}, 0)

	for _, entity := range bank.entities {
		if entityMatchesWithEntitySpecify(entity, exp) {
			matches = append(matches, entity)
		}
	}

	if len(matches) == 0 {
		return nil, fmt.Errorf("No entity found matching expression: %s", exp)
	}

	return matches, nil

}

func entityMatchesWithEntitySpecify(entity map[string]interface{}, exp Expression) bool {

	if exp.Header == ENTITY_SPECIFY {

		isEmpty := true
		allCorrect := true

		for key, value := range exp.Data {

			isEmpty = false

			if !slices.Contains([]string{"_id"}, key) &&
				!strings.EqualFold(entity[key].(string), value.(string)) {
				allCorrect = false
				break
			}

		}

		return !isEmpty && allCorrect

	}

	return false

}
