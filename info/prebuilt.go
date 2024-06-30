package info

import (
	"bestee/util"
)

var prebuiltExchangeBankInstance *PrebuiltExchangeBank

type PrebuiltExchangeBank struct {
	*pairMatcher
	data []exchangePair
}

func GetPrebuiltExchangeBankInstance() *PrebuiltExchangeBank {

	if prebuiltExchangeBankInstance == nil {
		prebuiltExchangeBankInstance = newBinaryExchangeBank()
	}

	return prebuiltExchangeBankInstance

}

func newBinaryExchangeBank() *PrebuiltExchangeBank {

	exePath := util.GetExcutableDir()
	config := util.ReadConfigJson()

	allExchangePairs := make([]exchangePair, 0)

	for _, partialPath := range config["prebuilt_exchange_pairs"].([]interface{}) {

		filePath := exePath + partialPath.(string)

		for _, pair := range util.ReadJsonIntoArrayOfObjects[exchangePair](filePath) {
			allExchangePairs = append(allExchangePairs, pair)
		}

	}

	return &PrebuiltExchangeBank{
		data:        allExchangePairs,
		pairMatcher: &pairMatcher{},
	}
}

func (bank *PrebuiltExchangeBank) FindResponse(query []string) ([]string, error) {
	return bank.pickBestMatch(query, bank.data)
}
