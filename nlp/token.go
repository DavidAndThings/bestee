package nlp

import (
	"strings"
)

func Tokenize(text string) []string {

	tokenDb := GetTokenDatabaseInstance()
	upperBound := len(text) + 1
	scores := make([]float64, upperBound)
	breakPoints := make([][]int, upperBound)

	scores[0] = 0
	breakPoints[0] = make([]int, 0)

	for i := range upperBound {

		if i == 0 {
			continue
		}

		localMax := -1.0
		localMaxPos := -1

		for j := range i {
			subtext := text[j:i]
			score := scores[j] + float64(len(subtext))*float64(tokenDb.HasToken(subtext))

			if score > float64(localMax) {
				localMax = score
				localMaxPos = j
			}
		}

		scores[i] = localMax
		breakPoints[i] = append(breakPoints[localMaxPos], localMaxPos)

	}

	// Obtain the tokens from the break points computed

	finalBreakPoints := append(breakPoints[upperBound-1], upperBound-1)
	tokens := extractTokensFromBreakPoints(text, finalBreakPoints)
	cleanTokens := cleanTokens(tokens)

	return cleanTokens

}

func extractTokensFromBreakPoints(text string, breakPoints []int) []string {

	tokens := make([]string, 0)
	previousPos := 0

	for _, pt := range breakPoints {

		if pt == 0 {
			continue
		}

		tokens = append(tokens, text[previousPos:pt])
		previousPos = pt

	}

	return tokens

}

func cleanTokens(tokens []string) []string {

	processedTokens := make([]string, 0)

	for _, t := range tokens {

		newToken := strings.TrimSpace(t)

		if newToken != "" {
			processedTokens = append(processedTokens, newToken)
		}

	}

	return processedTokens

}
