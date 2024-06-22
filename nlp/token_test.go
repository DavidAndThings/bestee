package nlp

import "testing"

func TestTokenizer(t *testing.T) {

	tokenizer := &Tokenizer{tokens: []string{"hello", "world"}}
	tokens := tokenizer.Run("hello    \t\tworld")

	if len(tokens) != 2 {
		t.Fail()
	}

}
