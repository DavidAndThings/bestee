package core

type Expression struct {
	header string
	data   map[string]interface{}
}

func (exp Expression) Evaluate(machine *Machine) {

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
