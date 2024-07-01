package nlp

import (
	"bestee/util"
	"strconv"
)

type Annotation struct {
	StartIndex int
	EndIndex   int
	Type       string
	Value      string
}

func (anno Annotation) HashStr() string {
	return strconv.Itoa(anno.StartIndex) +
		strconv.Itoa(anno.EndIndex) + anno.Type + anno.Value
}

type AnnotatedTokens struct {
	Tokens      []string
	Annotations *util.HashSet[Annotation]
}

func NewAnnotatedTokens(tokens []string) *AnnotatedTokens {
	return &AnnotatedTokens{
		Tokens:      tokens,
		Annotations: util.NewHashSet[Annotation](),
	}
}

func (tokens *AnnotatedTokens) HasAnnotationAt(start, end int, value string) bool {

	ans := false

	for _, annotation := range tokens.Annotations.Values() {

		if annotation.StartIndex == start &&
			annotation.EndIndex == end && annotation.Value == value {
			ans = true
			break
		}

	}

	return ans

}

func (tokens *AnnotatedTokens) AddAnnotations(annotations ...Annotation) {
	tokens.Annotations.AddAll(annotations...)
}

func (tokens *AnnotatedTokens) HasTopAnnotation(value string) bool {
	return tokens.HasAnnotationAt(0, tokens.Size(), value)
}

func (tokens *AnnotatedTokens) Size() int {
	return len(tokens.Tokens)
}
