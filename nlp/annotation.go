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

func NewAnnotatedTokens(text string) *AnnotatedTokens {
	return &AnnotatedTokens{
		Tokens:      Tokenize(text),
		Annotations: util.NewHashSet[Annotation](),
	}
}

func (self *AnnotatedTokens) HasAnnotationAt(start, end int, value string) bool {

	ans := false

	for _, annotation := range self.Annotations.Values() {

		if annotation.StartIndex == start &&
			annotation.EndIndex == end && annotation.Value == value {
			ans = true
			break
		}

	}

	return ans

}

func (self *AnnotatedTokens) AddAnnotations(annotations ...Annotation) {
	self.Annotations.AddAll(annotations...)
}

func (self *AnnotatedTokens) HasTopAnnotation(value string) bool {
	return self.HasAnnotationAt(0, self.Size(), value)
}

func (self *AnnotatedTokens) Size() int {
	return len(self.Tokens)
}

func (self *AnnotatedTokens) AlignWith(other *AnnotatedTokens) GlobalAlignmentResult {
	return GlobalSequencePairAlign(self.Tokens, other.Tokens)
}
