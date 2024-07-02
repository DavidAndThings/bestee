package core

import (
	"bestee/util"
	"fmt"
	"strings"

	"go.uber.org/zap"
)

var logger, _ = zap.NewProduction()
var sugarLogger = logger.Sugar()

type Machine struct {
	signalQueue *util.SynchronizedQueue[Signal]
	memory      *Memory
}

func NewMachineWithLogicBlocks(logicBlocks ...LogicBlock) *Machine {

	return &Machine{
		signalQueue: util.NewSynchronizedQueue[Signal](),
		memory:      newMemoryFromLogicBlocks(logicBlocks...),
	}

}

func (mach *Machine) AddToSignalQueue(newData ...Signal) {
	mach.signalQueue.Enqueue(newData...)
}

func (mach *Machine) RunEpoch() {

	for !mach.signalQueue.IsEmpty() {

		signal := mach.signalQueue.Pop()

		mach.processSignal(signal)
		mach.memory.Remember(signal)
		mach.AddToSignalQueue(mach.memory.Ruminate()...)

	}

}

func (mach *Machine) processSignal(signal Signal) {
	switch signal.Type {
	case BINARY_RESPONSE:
		fmt.Println(strings.Join(signal.Text.Tokens, " "))
	}
}
