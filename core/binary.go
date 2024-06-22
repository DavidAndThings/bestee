package core

import (
	"bestee/nlp"
	"bestee/util"
	"fmt"
	"strings"
)

var binaryExchangeBankInstance *BinaryExchangeBank

type BinaryExchangePair struct {
	input  []string
	output []string
}

func (pair BinaryExchangePair) matchesInput(query []string) bool {

	if len(query) != len(pair.input) {
		return false
	}

	ans := true

	for i := range len(query) {
		if strings.ToLower(query[i]) != strings.ToLower(pair.input[i]) {
			ans = false
			break
		}
	}

	return ans

}

type BinaryExchangeBank struct {
	data []BinaryExchangePair
}

func GetBinaryExchangeBankInstance() *BinaryExchangeBank {

	if binaryExchangeBankInstance == nil {
		binaryExchangeBankInstance = newBinaryExchangeBank()
	}

	return binaryExchangeBankInstance

}

func newBinaryExchangeBank() *BinaryExchangeBank {

	exePath := util.GetExcutableDir()
	config := util.ReadConfigJson()
	tokenizer := nlp.GetTokenizerInstance()

	allExchangePairs := make([]BinaryExchangePair, 0)

	for _, partialPath := range config["binary_exchange_pairs"].([]interface{}) {

		filePath := exePath + partialPath.(string)

		for _, doc := range util.ReadJsonIntoArray(filePath) {

			allExchangePairs = append(
				allExchangePairs,
				BinaryExchangePair{
					input:  tokenizer.Run(doc["input"].(string)),
					output: tokenizer.Run(doc["output"].(string)),
				},
			)

		}

	}

	return &BinaryExchangeBank{data: allExchangePairs}
}

func (bank *BinaryExchangeBank) Process(machine *Machine) []Expression {

	newExps := make([]Expression, 0)

	for id, tokenizedText := range machine.findUnmatchedTokenizedText() {

		response, err := bank.findResponse(tokenizedText.Data["text"].([]string))

		if err == nil {
			newExps = append(newExps, BuildResponseFromBinary(id, response))
		}

	}

	return newExps

}

func (bank *BinaryExchangeBank) findResponse(query []string) ([]string, error) {

	matchFound := false
	var resp []string

	for _, pair := range bank.data {
		if pair.matchesInput(query) {

			resp = pair.output
			matchFound = true
			break

		}
	}

	if !matchFound {
		return nil, fmt.Errorf("No match found for input: %s", query)
	}

	return resp, nil

}
