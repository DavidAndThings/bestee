package nlp

import (
	"testing"
)

func TestTokenizer(t *testing.T) {

	tokens := Tokenize("hello    \t\tworld")

	if len(tokens) != 2 {
		t.Fail()
	}

}
