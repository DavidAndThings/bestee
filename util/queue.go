package util

import (
	"sync"
)

type SynchronizedQueue[T any] struct {
	data []T
	lock sync.Mutex
}

func NewSynchronizedQueue[T any]() *SynchronizedQueue[T] {

	return &SynchronizedQueue[T]{
		data: make([]T, 0),
	}

}

func (queue *SynchronizedQueue[T]) Enqueue(items ...T) {

	queue.lock.Lock()
	queue.data = append(queue.data, items...)
	queue.lock.Unlock()

}

func (queue *SynchronizedQueue[T]) Pop() T {

	queue.lock.Lock()
	ans := queue.data[0]
	queue.data = queue.data[1:]
	queue.lock.Unlock()
	return ans

}

func (queue *SynchronizedQueue[T]) IsEmpty() bool {
	return len(queue.data) == 0
}

func (queue *SynchronizedQueue[T]) Size() int {
	return len(queue.data)
}
