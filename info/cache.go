package info

type refLookAsideCache struct {
	first  string
	second string
	third  string
	other  []string
}

func (cache *refLookAsideCache) computeExchangePairs(bank *EntityBank) []exchangePair {
	return make([]exchangePair, 0)
}
