package core

import (
	"go.uber.org/zap"
)

var logger, _ = zap.NewProduction()
var sugarLogger = logger.Sugar()

type Machine struct {
	Memory      *ExpressionArray `json:"memory"`
	SignalQueue *ExpressionArray `json:"signal_queue"`
	logic       []LogicBlock
}

func NewMachineWithLogicBlocks(logicBlocks ...LogicBlock) *Machine {
	return &Machine{
		Memory:      NewExpressionArray(),
		SignalQueue: NewExpressionArray(),
		logic:       logicBlocks,
	}
}

func (mach *Machine) AddToInputQueue(newData ...Expression) {
	mach.SignalQueue.Add(newData...)
}

func (mach *Machine) RunEpoch() {

	previousSize := -1

	for mach.Memory.Size() > previousSize {

		previousSize = mach.Memory.Size()
		mach.increment()

	}

}

func (mach *Machine) increment() {

	for _, block := range mach.logic {
		mach.AddToInputQueue(block.Process(mach)...)
	}

	for !mach.SignalQueue.IsEmpty() {

		exp := mach.SignalQueue.Pop()
		exp.Evaluate(mach)

		sugarLogger.Infow(
			"Machine State",
			"expression_being_processed", exp,
			"memory", mach.Memory.data,
			"signal_queue", mach.SignalQueue.data,
		)
	}

}

func (mach *Machine) findUntranslatedSpecifications() map[string]Expression {

	entitySpecifications := make(map[string]Expression)

	for _, exp := range mach.Memory.data {

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
