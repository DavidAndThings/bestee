package core

import (
	"bestee/nlp"

	"github.com/google/uuid"
)

const ANNOTATED_TEXT = "ANNOTATED_TEXT"
const BINARY_RESPONSE = "BINARY_RESPONSE"

type Signal struct {
	Type string
	ID   string
	From string
	To   string
	For  string
	Text *nlp.AnnotatedTextSequence
}

func (signal Signal) HashStr() string {
	return signal.ID
}

func BuildAnnotatedText(text string) Signal {

	return Signal{
		Type: ANNOTATED_TEXT,
		ID:   uuid.New().String(),
		Text: nlp.NewAnnotatedTextSequence(text),
	}

}

func BuildBinaryResponse(respFor string, resp *nlp.AnnotatedTextSequence) Signal {

	return Signal{
		Type: BINARY_RESPONSE,
		For:  respFor,
		Text: resp,
	}

}
