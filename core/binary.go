package core

import "bestee/util"

type BinaryExchangePair struct {
	input  []string
	output []string
}

type BinaryExchangeBank struct {
	data []BinaryExchangePair
}

func NewBinaryExchangeBank() *BinaryExchangeBank {
	return &BinaryExchangeBank{data: make([]BinaryExchangePair, 0)}
}

func (bank *BinaryExchangeBank) AddFromJsonFile(filePath string) {

	for _, doc := range util.ReadJsonIntoArray(filePath) {

		bank.data = append(
			bank.data,
			BinaryExchangePair{
				input:  doc["input"].([]string),
				output: doc["output"].([]string),
			},
		)

	}

}
