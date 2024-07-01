package info

import (
	"bestee/nlp"
	"encoding/json"
	"math"
	"strings"
)

type TokenArray []string

func (t *TokenArray) UnmarshalJSON(b []byte) error {
	trimmedStr := strings.Trim(string(b), "\"")
	*t = nlp.Tokenize(trimmedStr)
	return nil
}

func (t TokenArray) MarshalJSON() ([]byte, error) {
	return json.Marshal(strings.Join(t, " "))
}

type ExchangePair struct {
	Input      TokenArray `json:"input"`
	Output     TokenArray `json:"output"`
	SourceType string     `json:"source_type"`
	SourceID   string
}

func (pair ExchangePair) matchScore(query []string) float64 {
	return nlp.GlobalSequencePairAlign(pair.Input, query).Score
}

type pairMatcher struct {
}

func (matcher *pairMatcher) pickBestMatch(
	query []string, exchangePairs []ExchangePair) (ExchangePair, error) {

	var bestMatch ExchangePair
	highestScore := math.Inf(-1)

	for _, pair := range exchangePairs {
		matchScore := pair.matchScore(query)

		if matchScore > highestScore {
			highestScore = matchScore
			bestMatch = pair
		}
	}

	return bestMatch, nil

}
