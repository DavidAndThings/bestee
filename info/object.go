package info

import (
	"bestee/nlp"
	"bestee/util"
	"fmt"
)

var entityBankInstance *ObjectBank

type ObjectBank struct {
	*pairMatcher
	cache  *refLookAsideCache
	people *util.HashSet[*person]
}

func GetEntityBankInstance() *ObjectBank {

	if entityBankInstance == nil {
		entityBankInstance = newEntityBank()
	}

	return entityBankInstance

}

func newEntityBank() *ObjectBank {

	exePath := util.GetExcutableDir()
	config := util.ReadConfigJson()

	allPeople := util.NewHashSet[*person]()

	for _, partialPath := range config["people"].([]interface{}) {

		filePath := exePath + partialPath.(string)
		allPeople.AddAll(util.ReadJsonIntoArrayOfObjects[*person](filePath)...)

	}

	return &ObjectBank{
		pairMatcher: &pairMatcher{},
		cache:       &refLookAsideCache{},
		people:      allPeople,
	}

}

func (bank *ObjectBank) FindResponse(query *nlp.AnnotatedTextSequence) (ExchangePair, error) {

	var resp ExchangePair
	allPairs := bank.getAvailableExchangePairs()

	if len(allPairs) == 0 {
		return resp, fmt.Errorf("No exchange pair is available!")
	}

	return bank.pickBestMatch(query, allPairs)

}

func (bank *ObjectBank) getAvailableExchangePairs() []ExchangePair {

	allPairs := bank.cache.computeExchangePairs(bank)

	for _, person := range bank.people.Values() {
		allPairs = append(allPairs, person.computeExchangePairs(bank)...)
	}

	return allPairs

}
