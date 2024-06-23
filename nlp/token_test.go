package nlp

import (
	"bestee/util"
	"testing"
)

func TestTokenizer(t *testing.T) {

	tokenStorage := util.NewStrHashStore()
	tokenStorage.AddAll("hello", "world")

	tokenizer := &Tokenizer{tokens: tokenStorage}
	tokens := tokenizer.Run("hello    \t\tworld")

	if len(tokens) != 2 {
		t.Fail()
	}

}
