package core

type Machine struct {
	array *ExpressionArray
	queue *ExpressionQueue
	logic []LogicBlock
}

func NewMachineWithLogicBlocks(logicBlocks ...LogicBlock) *Machine {
	return &Machine{array: NewExpressionArray(), queue: NewExpressionQueue(), logic: logicBlocks}
}

func (mach *Machine) AddToQueue(newData ...*Expression) {
	mach.queue.Add(newData...)
}

func (mach *Machine) Run() {

	previousSize := -1

	for mach.array.Size() > previousSize {

		previousSize = mach.array.Size()
		mach.increment()

	}

}

func (mach *Machine) increment() {

	for _, block := range mach.logic {
		block.Process(mach)

	for !mach.queue.IsEmpty() {
		exp := mach.queue.Pop()
		exp.Evaluate(mach.array)
	}

}
