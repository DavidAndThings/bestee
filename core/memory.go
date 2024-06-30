package core

type Memory struct {
	shortTerm []Signal
	longTerm  []LogicBlock
}

func newMemoryFromLogicBlocks(logicBlocks ...LogicBlock) *Memory {
	return &Memory{shortTerm: make([]Signal, 0), longTerm: logicBlocks}
}

func (mem *Memory) Remember(signal Signal) {
	mem.shortTerm = append(mem.shortTerm, signal)
}

func (mem *Memory) Ruminate() []Signal {

	ideas := make([]Signal, 0)

	for _, block := range mem.longTerm {
		ideas = append(ideas, block.Process(mem)...)
	}

	return ideas

}

func (mem *Memory) runQuery(query func(*Memory) []Signal) []Signal {
	return query(mem)
}
