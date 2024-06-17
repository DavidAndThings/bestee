package core

const ENTITY_SPECIFY = "ENTITY_SPECIFY"
const ENTITY_TRANSLATE = "ENTITY_TRANSLATE"
const ADD_INSTR = "ADD_INSTR"

type Expression struct {
	header string
	data   map[string]interface{}
}

func (exp Expression) Evaluate(machine *Machine) {
	switch exp.header {
	case ADD_INSTR:
		machine.memory.Add(exp.data["to_add"].(Expression))
	}
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

func buildAddInstr(exp Expression) Expression {

	return Expression{
		header: ADD_INSTR,
		data: map[string]interface{}{
			"to_add": exp,
		},
	}

}

func buildEntityTranslate(from string, to string) Expression {

	return Expression{
		header: ENTITY_TRANSLATE,
		data: map[string]interface{}{
			"from": from,
			"to":   to,
		},
	}

}
