package core

import (
	"bestee/nlp"

	"github.com/google/uuid"
)

const ENTITY_SPECIFY = "ENTITY_SPECIFY"
const ENTITY_TRANSLATE = "ENTITY_TRANSLATE"
const TOKENIZED_TEXT = "TOKENIZED_TEXT"
const RESPONSE_FROM_BINARY = "RESPONSE_FROM_BINARY"

type Expression struct {
	Header string                 `json:"header"`
	Data   map[string]interface{} `json:"data"`
}

func (exp Expression) GetDataString(key string) string {
	return exp.Data[key].(string)
}

func (exp Expression) GetDataExpression(key string) Expression {
	return exp.Data[key].(Expression)
}

func BuildEntityTranslate(from string, to string) Expression {

	return Expression{
		Header: ENTITY_TRANSLATE,
		Data: map[string]interface{}{
			"from": from,
			"to":   to,
		},
	}

}

func BuildTokenizedText(text string) Expression {

	return Expression{
		Header: TOKENIZED_TEXT,
		Data: map[string]interface{}{
			"_id":  uuid.New().String(),
			"text": nlp.Tokenize(text),
		},
	}

}

func BuildResponseFromBinary(respFor string, resp []string) Expression {

	return Expression{
		Header: RESPONSE_FROM_BINARY,
		Data: map[string]interface{}{
			"for":  respFor,
			"resp": resp,
		},
	}

}
