package core

const ENTITY_SPECIFY = "ENTITY_SPECIFY"
const ENTITY_TRANSLATE = "ENTITY_TRANSLATE"
const ADD_INSTR = "ADD_INSTR"

type Expression struct {
	Header string                 `json:"header"`
	Data   map[string]interface{} `json:"data"`
}

func (exp Expression) Evaluate(machine *Machine) {
	switch exp.Header {
	case ADD_INSTR:
		machine.Memory.Add(exp.GetDataExpression("to_add"))
	}
}

func (exp Expression) GetDataString(key string) string {
	return exp.Data[key].(string)
}

func (exp Expression) GetDataExpression(key string) Expression {
	return exp.Data[key].(Expression)
}

type ExpressionArray struct {
	data []Expression
}

func NewExpressionArray() *ExpressionArray {
	return &ExpressionArray{data: make([]Expression, 0)}
}

func (array *ExpressionArray) IsEmpty() bool {
	return len(array.data) == 0
}

func (array *ExpressionArray) Pop() Expression {
	node := array.data[0]
	array.data = array.data[1:]
	return node
}

func (array *ExpressionArray) Add(newData ...Expression) {
	array.data = append(array.data, newData...)
}

func (array *ExpressionArray) Size() int {
	return len(array.data)
}

func BuildAddInstr(exp Expression) Expression {

	return Expression{
		Header: ADD_INSTR,
		Data: map[string]interface{}{
			"to_add": exp,
		},
	}

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
