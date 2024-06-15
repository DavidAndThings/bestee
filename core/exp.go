package core

type Expression struct {
	header string
	data   []interface{}
}

func (exp *Expression) Evaluate(machine *Machine) {

}

type Pair struct {
	key   string
	value interface{}
}

type ExpressionArray struct {
	data []*Expression
}

func NewExpressionArray() *ExpressionArray {
	return &ExpressionArray{data: make([]*Expression, 0)}
}

func (array *ExpressionArray) Size() int {
	return len(array.data)
}

func (array *ExpressionArray) GetAt(index int) *Expression {
	return array.data[index]
}

func (array *ExpressionArray) Add(newData ...*Expression) {
	array.data = append(array.data, newData...)
}

type ExpressionQueue struct {
	data []*Expression
}

func NewExpressionQueue() *ExpressionQueue {
	return &ExpressionQueue{data: make([]*Expression, 0)}
}

func (queue *ExpressionQueue) IsEmpty() bool {
	return len(queue.data) == 0
}

func (queue *ExpressionQueue) Pop() *Expression {
	node := queue.data[0]
	queue.data = queue.data[1:]
	return node
}

func (queue *ExpressionQueue) Add(newData ...*Expression) {
	queue.data = append(queue.data, newData...)
}
