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

type exchangePair struct {
	Input  TokenArray `json:"input"`
	Output TokenArray `json:"output"`
}

func (pair exchangePair) matchScore(query []string) float64 {
	return nlp.GlobalSequencePairAlign(pair.Input, query).Score
}

type pairMatcher struct {
}

func (matcher *pairMatcher) pickBestMatch(
	query []string, exchangePairs []exchangePair) ([]string, error) {

	var bestMatch exchangePair
	highestScore := math.Inf(-1)

	for _, pair := range exchangePairs {
		matchScore := pair.matchScore(query)

		if matchScore > highestScore {
			highestScore = matchScore
			bestMatch = pair
		}
	}

	return bestMatch.Output, nil

}
