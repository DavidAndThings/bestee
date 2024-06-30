package core

import (
	"bestee/info"
	"bestee/nlp"
)

type BinaryExchangeBlock struct {
	entities *info.EntityBank
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
	unmatchedPlainText := memory.runQuery(findUnmatchedPlainText)

	for _, sig := range unmatchedPlainText {
		newExps = bank.processPlainText(sig)
	}

	return newExps

}

func (bank *BinaryExchangeBlock) processPlainText(signal Signal) []Signal {

	ans := make([]Signal, 0)
	tokenizedInput := nlp.Tokenize(signal.Text)
	resp, err := bank.prebuilt.FindResponse(tokenizedInput)

	if err == nil {
		ans = append(ans, BuildBinaryResponse(signal.ID, resp))
	}

	return ans

}
