package info

import (
	"bestee/nlp"
	"math"
)

type ExchangePair struct {
	Input      *nlp.AnnotatedTextSequence `json:"input"`
	Output     *nlp.AnnotatedTextSequence `json:"output"`
	SourceType string                     `json:"source_type"`
	SourceID   string
}

func (pair ExchangePair) matchScore(query *nlp.AnnotatedTextSequence) float64 {
	return pair.Input.SimilarityScore(query)
}

type pairMatcher struct {
}

func (matcher *pairMatcher) pickBestMatch(
	query *nlp.AnnotatedTextSequence, exchangePairs []ExchangePair) (ExchangePair, error) {

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
