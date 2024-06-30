package core

import (
	"strings"

	"github.com/google/uuid"
)

const PLAIN_TEXT = "PLAIN_TEXT"
const BINARY_RESPONSE = "BINARY_RESPONSE"

type Signal struct {
	Type string
	ID   string
	From string
	To   string
	For  string
	Text string
}

func (signal Signal) HashStr() string {
	return signal.ID
}

func BuildPlainText(text string) Signal {

	return Signal{
		Type: PLAIN_TEXT,
		ID:   uuid.New().String(),
		Text: text,
	}

}

func BuildBinaryResponse(respFor string, resp []string) Signal {

	return Signal{
		Type: BINARY_RESPONSE,
		For:  respFor,
		Text: strings.Join(resp, " "),
	}

}
