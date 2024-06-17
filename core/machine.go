package core

type Machine struct {
	memory *ExpressionArray
	queue  *ExpressionArray
	logic  []LogicBlock
}

func NewMachineWithLogicBlocks(logicBlocks ...LogicBlock) *Machine {
	return &Machine{memory: NewExpressionArray(), queue: NewExpressionArray(), logic: logicBlocks}
}

func (mach *Machine) AddToQueue(newData ...Expression) {
	mach.queue.Add(newData...)
}

func (mach *Machine) Run() {

	previousSize := -1

	for mach.memory.Size() > previousSize {

		previousSize = mach.memory.Size()
		mach.increment()

	}

}

func (mach *Machine) increment() {

	for _, block := range mach.logic {
		block.Process(mach)
	}

	for !mach.queue.IsEmpty() {
		exp := mach.queue.Pop()
		exp.Evaluate(mach)
	}

}
