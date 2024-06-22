package core

const ENTITY_SPECIFY = "ENTITY_SPECIFY"
const ENTITY_TRANSLATE = "ENTITY_TRANSLATE"

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
