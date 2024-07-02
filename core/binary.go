package core

import (
	"bestee/info"
)

type BinaryExchangeBlock struct {
	entities *info.ObjectBank
	prebuilt *info.PrebuiltExchangeBank
}

func NewBinaryExchangeBlock() *BinaryExchangeBlock {

	return &BinaryExchangeBlock{
		entities: info.GetEntityBankInstance(),
		prebuilt: info.GetPrebuiltExchangeBankInstance(),
	}

}

func (bank *BinaryExchangeBlock) Process(memory *Memory) []Signal {

	newExps := make([]Signal, 0)
	unmatchedPlainText := memory.runQuery(findUnmatchedAnnotatedText)

	for _, sig := range unmatchedPlainText {
		newExps = bank.processPlainText(sig)
	}

	return newExps

}

func (bank *BinaryExchangeBlock) processPlainText(signal Signal) []Signal {

	ans := make([]Signal, 0)
	resp, err := bank.prebuilt.FindResponse(signal.Text)

	if err == nil {
		ans = append(ans, BuildBinaryResponse(signal.ID, resp.Output))
	}

	return ans

}
