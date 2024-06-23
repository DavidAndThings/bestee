package util

type StrHashStore struct {
	data map[string]string
}

func NewStrHashStore() *StrHashStore {
	return &StrHashStore{data: make(map[string]string)}
}

func (store *StrHashStore) Add(newStr string) {
	store.data[GetMD5Hash(newStr)] = newStr
}

func (store *StrHashStore) AddAll(newStrs ...string) {

	for _, i := range newStrs {
		store.Add(i)
	}

}

func (store *StrHashStore) Contains(queryStr string) bool {

	_, ok := store.data[GetMD5Hash(queryStr)]
	return ok

}

func (store *StrHashStore) Remove(queryStr string) {
	delete(store.data, GetMD5Hash(queryStr))
}

func (store *StrHashStore) Size() int {
	return len(store.data)
}
