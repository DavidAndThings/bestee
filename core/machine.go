package core

type Machine struct {
	memory      *ExpressionArray
	signalQueue *ExpressionArray
	logic       []LogicBlock
}

func NewMachineWithLogicBlocks(logicBlocks ...LogicBlock) *Machine {
	return &Machine{
		memory:      NewExpressionArray(),
		signalQueue: NewExpressionArray(),
		logic:       logicBlocks,
	}
}

func (mach *Machine) AddToInputQueue(newData ...Expression) {
	mach.signalQueue.Add(newData...)
}

func (mach *Machine) RunEpoch() {

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

	for !mach.signalQueue.IsEmpty() {
		exp := mach.signalQueue.Pop()
		exp.Evaluate(mach)
	}

}

func (mach *Machine) findUntranslatedSpecifications() map[string]Expression {

	entitySpecifications := make(map[string]Expression)

	for _, exp := range mach.memory.data {

		switch exp.Header {
		case ENTITY_SPECIFY:
			entitySpecifications[exp.Data["_id"].(string)] = exp

		case ENTITY_TRANSLATE:
			if _, ok := entitySpecifications[exp.Data["from"].(string)]; ok {
				delete(entitySpecifications, exp.Data["from"].(string))
			}
		}

	}

	return entitySpecifications

}
