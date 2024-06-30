package core

type LogicBlock interface {
	Process(memory *Memory) []Signal
}
