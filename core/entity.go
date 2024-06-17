package core

import (
	"bestee/util"
	"errors"
	"fmt"
	"slices"
)

const _ENTITY_SPECIFY = "ENTITY_SPECIFY"
const _ENTITY_TRANSLATE = "ENTITY_TRANSLATE"

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

	entitySpecifications := make(map[string]Expression)

	for _, exp := range machine.memory.data {

		switch exp.header {
		case _ENTITY_SPECIFY:
			entitySpecifications[exp.data["_id"].(string)] = exp

		case _ENTITY_TRANSLATE:
			if _, ok := entitySpecifications[exp.data["from"].(string)]; ok {
				delete(entitySpecifications, exp.data["from"].(string))
			}
		}

	}

	for id, specification := range entitySpecifications {

		entityMatches, searchErr := bank.findWithEntitySpecify(specification)

		if searchErr != nil {
			for _, match := range entityMatches {
				machine.AddToQueue(
					Expression{
						header: "ADD_INSTR",
						data: map[string]interface{}{
							"to_add": Expression{
								header: _ENTITY_TRANSLATE,
								data: map[string]interface{}{
									"from": id,
									"to":   match["id"].(string),
								},
							},
						},
					},
				)
			}
		}

	}

}

func (bank *EntityBank) findWithEntitySpecify(exp Expression) ([]map[string]interface{}, error) {

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

	if exp.header == _ENTITY_SPECIFY {

		isEmpty := true
		allCorrect := true

		for key, value := range exp.data {

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
