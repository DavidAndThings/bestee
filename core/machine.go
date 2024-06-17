package core

type Machine struct {
	memory      *ExpressionArray
	inputQueue  *ExpressionArray
	outputQueue *ExpressionArray
	logic       []LogicBlock
}

func NewMachineWithLogicBlocks(logicBlocks ...LogicBlock) *Machine {
	return &Machine{
		memory:      NewExpressionArray(),
		inputQueue:  NewExpressionArray(),
		outputQueue: NewExpressionArray(),
		logic:       logicBlocks,
	}
}

func (mach *Machine) AddToInputQueue(newData ...Expression) {
	mach.inputQueue.Add(newData...)
}

func (mach *Machine) RunEpoch() {

	previousSize := -1

	for mach.memory.Size() > previousSize {

		previousSize = mach.memory.Size()
		mach.increment()

	}

}

func (mach *Machine) OutputQueueIsEmpty() bool {
	return mach.outputQueue.IsEmpty()
}

func (mach *Machine) PopOutputQueue() Expression {
	return mach.outputQueue.Pop()
}

func (mach *Machine) increment() {

	for _, block := range mach.logic {
		block.Process(mach)
	}

	for !mach.inputQueue.IsEmpty() {
		exp := mach.inputQueue.Pop()
		exp.Evaluate(mach)
	}

}

func (mach *Machine) findUntranslatedSpecifications() map[string]Expression {

	entitySpecifications := make(map[string]Expression)

	for _, exp := range mach.memory.data {

		switch exp.header {
		case ENTITY_SPECIFY:
			entitySpecifications[exp.data["_id"].(string)] = exp

		case ENTITY_TRANSLATE:
			if _, ok := entitySpecifications[exp.data["from"].(string)]; ok {
				delete(entitySpecifications, exp.data["from"].(string))
			}
		}

	}

	return entitySpecifications

}
