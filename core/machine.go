package core

import (
	"sync"

	"go.uber.org/zap"
)

var logger, _ = zap.NewProduction()
var sugarLogger = logger.Sugar()

type Machine struct {
	array     []Expression
	counter   int
	arrayLock sync.Mutex
	logic     LogicBlock
}

func NewMachineWithLogicBlocks(logicBlock LogicBlock) *Machine {
	return &Machine{
		array:   make([]Expression, 0),
		counter: 0,
		logic:   logicBlock,
	}
}

func (mach *Machine) AddToSignalQueue(newData ...Expression) {

	mach.arrayLock.Lock()
	mach.array = append(mach.array, newData...)
	mach.arrayLock.Unlock()

}

func (mach *Machine) RunEpoch() {

	previousCounterValue := -1

	for mach.counter > previousCounterValue {

		previousCounterValue = mach.counter
		mach.increment()

	}

}

func (mach *Machine) increment() {

	mach.runLogic()
	mach.arrayLock.Lock()

	for _, exp := range mach.getAfterCounter() {

		sugarLogger.Infow(
			"Machine State",
			"expression_being_processed", exp,
		)

	}

	mach.counter = len(mach.array)
	mach.arrayLock.Unlock()

}

func (mach *Machine) getBeforeCounter() []Expression {
	return mach.array[0:mach.counter]
}

func (mach *Machine) getAfterCounter() []Expression {
	return mach.array[mach.counter:len(mach.array)]
}

func (mach *Machine) runLogic() {

	mach.arrayLock.Lock()
	mach.array = append(mach.array, mach.logic.Process(mach)...)
	mach.arrayLock.Unlock()

}

func (mach *Machine) findUntranslatedSpecifications() map[string]Expression {

	entitySpecifications := make(map[string]Expression)

	for _, exp := range mach.getBeforeCounter() {

		switch exp.Header {
		case ENTITY_SPECIFY:
			entitySpecifications[exp.Data["_id"].(string)] = exp

		case ENTITY_TRANSLATE:
			delete(entitySpecifications, exp.Data["from"].(string))
		}

	}

	return entitySpecifications

}
