package info

import (
	"bestee/util"
	"fmt"
)

var entityBankInstance *EntityBank

type EntityBank struct {
	*pairMatcher
	cache  *refLookAsideCache
	people *util.HashSet[*person]
}

func GetEntityBankInstance() *EntityBank {

	if entityBankInstance == nil {
		entityBankInstance = newEntityBank()
	}

	return entityBankInstance

}

func newEntityBank() *EntityBank {

	exePath := util.GetExcutableDir()
	config := util.ReadConfigJson()

	allPeople := util.NewHashSet[*person]()

	for _, partialPath := range config["people"].([]interface{}) {

		filePath := exePath + partialPath.(string)
		allPeople.AddAll(util.ReadJsonIntoArrayOfObjects[*person](filePath)...)

	}

	return &EntityBank{
		pairMatcher: &pairMatcher{},
		cache:       &refLookAsideCache{},
		people:      allPeople,
	}

}

func (bank *EntityBank) FindResponse(query []string) ([]string, error) {

	allPairs := bank.getAvailableExchangePairs()

	if len(allPairs) == 0 {
		return nil, fmt.Errorf("No exchange pair is available!")
	}

	return bank.pickBestMatch(query, allPairs)

}

func (bank *EntityBank) getAvailableExchangePairs() []exchangePair {

	allPairs := bank.cache.computeExchangePairs(bank)

	for _, person := range bank.people.Values() {
		allPairs = append(allPairs, person.computeExchangePairs(bank)...)
	}

	return allPairs

}
