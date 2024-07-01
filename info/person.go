package info

import "bestee/util"

type person struct {
	ID         string        `json:"_id"`
	Sex        string        `json:"sex"`
	FirstName  string        `json:"first_name"`
	MiddleName string        `json:"last_name"`
	LastName   string        `json:"middle_name"`
	BirthDate  util.JsonDate `json:"birth_date"`
}

func (person *person) HashStr() string {
	return person.ID
}

func (person *person) computeExchangePairs(bank *ObjectBank) []ExchangePair {
	return make([]ExchangePair, 0)
}
