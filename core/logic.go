package core

type LogicBlock interface {
	Process(array *ExpressionArray, queue *ExpressionQueue)
}
