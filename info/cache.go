package info

type refLookAsideCache struct {
	first  string
	second string
	third  string
	other  []string
}

func (cache *refLookAsideCache) computeExchangePairs(bank *ObjectBank) []ExchangePair {
	return make([]ExchangePair, 0)
}
