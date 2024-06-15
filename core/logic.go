package core

type LogicBlock interface {
	Process(machine *Machine)
}
