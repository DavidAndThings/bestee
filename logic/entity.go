package logic

import "bestee/util"

type EntityBank struct {
	entities []map[string]interface{}
}

func NewEmptyEntityBank() *EntityBank {
	return &EntityBank{entities: make([]map[string]interface{}, 0)}
}

func (bank *EntityBank) AddFromJsonFile(filePath string) {
	bank.entities = append(bank.entities, util.ReadJsonIntoArray(filePath)...)
}
