package core

import "bestee/util"

func findUnmatchedPlainText(mem *Memory) []Signal {

	storage := util.NewHashSet[Signal]()

	for _, sig := range mem.shortTerm {

		switch sig.Type {
		case PLAIN_TEXT:
			storage.Add(sig)
		case BINARY_RESPONSE:

			if storage.HasItemWithHash(sig.For) {
				match := storage.GetWithHash(sig.For)
				storage.Remove(match)
			}

		}

	}

	return storage.Values()

}
