package util

type Hashable interface {
	HashStr() string
}

type HashSet[T Hashable] struct {
	data map[string]T
}

func NewHashSet[T Hashable]() *HashSet[T] {

	return &HashSet[T]{
		data: make(map[string]T),
	}

}

func (set *HashSet[T]) Add(item T) {
	set.data[item.HashStr()] = item
}

func (set *HashSet[T]) AddAll(items ...T) {

	for _, item := range items {
		set.Add(item)
	}

}

func (set *HashSet[T]) Values() []T {

	ans := make([]T, 0)

	for _, v := range set.data {
		ans = append(ans, v)
	}

	return ans

}

func (set *HashSet[T]) GetWithHash(key string) T {
	return set.data[key]
}

func (set *HashSet[T]) Contains(item T) bool {
	_, ok := set.data[item.HashStr()]
	return ok
}

func (set *HashSet[T]) HasItemWithHash(hashStr string) bool {
	_, ok := set.data[hashStr]
	return ok
}

func (set *HashSet[T]) Remove(item T) {
	delete(set.data, item.HashStr())
}

func (set *HashSet[T]) Size() int {
	return len(set.data)
}
