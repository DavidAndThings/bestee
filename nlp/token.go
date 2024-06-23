package nlp

import (
	"bestee/util"
	"strings"

	"go.uber.org/zap"
)

var logger, _ = zap.NewProduction()
var sugarLogger = logger.Sugar()

var tokenizerInstance *Tokenizer

type Tokenizer struct {
	tokens *util.StrHashStore
}

func newTokenizer() *Tokenizer {

	exeDir := util.GetExcutableDir()
	config := util.ReadConfigJson()
	allTokens := util.NewStrHashStore()

	for _, tokenFilePath := range config["predefined_token_files"].([]interface{}) {

		tokens, err := util.ReadIntoStrArray(exeDir + tokenFilePath.(string))

		if err == nil {
			allTokens.AddAll(tokens...)
		}

	}

	return &Tokenizer{tokens: allTokens}

}

func GetTokenizerInstance() *Tokenizer {

	if tokenizerInstance == nil {
		tokenizerInstance = newTokenizer()
	}

	return tokenizerInstance

}

func (tokenizer *Tokenizer) Run(text string) []string {

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
			score := scores[j] + float64(len(subtext))*float64(tokenizer.Contains(subtext))

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

	sugarLogger.Infow(
		"Tokenization Summary",
		"Scores", scores,
		"Break_Points", breakPoints[upperBound-1],
		"Tokens", cleanTokens,
	)

	return cleanTokens

}

func (tokenizer *Tokenizer) Contains(token string) int {

	ans := 0

	if tokenizer.tokens.Contains(token) {
		ans = 1
	}

	return ans

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
