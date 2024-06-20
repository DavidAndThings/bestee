package core

import (
	"bestee/util"
	"errors"
	"fmt"
	"slices"
)

type EntityBank struct {
	entities []map[string]interface{}
}

func NewEmptyEntityBank() *EntityBank {
	return &EntityBank{entities: make([]map[string]interface{}, 0)}
}

func (bank *EntityBank) AddFromJsonFile(filePath string) {
	bank.entities = append(bank.entities, util.ReadJsonIntoArray(filePath)...)
}

func (bank *EntityBank) Process(machine *Machine) {

	for id, specification := range machine.findUntranslatedSpecifications() {

		entityMatches, searchErr := bank.findEntitiesWithEntitySpecify(specification)

		if searchErr == nil {
			for _, match := range entityMatches {
				machine.AddToInputQueue(
					BuildAddInstr(BuildEntityTranslate(id, match["_id"].(string))),
				)
			}
		}

	}

}

func (bank *EntityBank) findEntitiesWithEntitySpecify(exp Expression) ([]map[string]interface{}, error) {

	matches := make([]map[string]interface{}, 0)

	for _, entity := range bank.entities {
		if entityMatchesWithEntitySpecify(entity, exp) {
			matches = append(matches, entity)
		}
	}

	if len(matches) == 0 {
		return nil, errors.New(fmt.Sprintf("No entity found matching expression: %s", exp))
	}

	return matches, nil

}

func entityMatchesWithEntitySpecify(entity map[string]interface{}, exp Expression) bool {

	if exp.Header == ENTITY_SPECIFY {

		isEmpty := true
		allCorrect := true

		for key, value := range exp.Data {

			isEmpty = false

			if !slices.Contains([]string{"_id"}, key) && entity[key] != value {
				allCorrect = false
				break
			}

		}

		return !isEmpty && allCorrect

	}

	return false

}
