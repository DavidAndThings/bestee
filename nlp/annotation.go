package nlp

import (
	"bestee/util"
	"encoding/json"
	"strconv"
	"strings"
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

type AnnotatedTextSequence struct {
	Tokens      []string
	Annotations *util.HashSet[Annotation]
}

func NewAnnotatedTextSequence(text string) *AnnotatedTextSequence {
	return &AnnotatedTextSequence{
		Tokens:      Tokenize(text),
		Annotations: util.NewHashSet[Annotation](),
	}
}

func (self *AnnotatedTextSequence) HasAnnotationAt(start, end int, value string) bool {

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

func (self *AnnotatedTextSequence) GetAnnotationsWithType(queryType string) []Annotation {

	ans := make([]Annotation, 0)

	for _, annotation := range self.Annotations.Values() {
		if annotation.Type == queryType {
			ans = append(ans, annotation)
		}
	}

	return ans

}

func (self *AnnotatedTextSequence) AddAnnotations(annotations ...Annotation) {
	self.Annotations.AddAll(annotations...)
}

func (self *AnnotatedTextSequence) HasTopAnnotation(value string) bool {
	return self.HasAnnotationAt(0, self.Size(), value)
}

func (self *AnnotatedTextSequence) Size() int {
	return len(self.Tokens)
}

func (self *AnnotatedTextSequence) AlignWith(other *AnnotatedTextSequence) GlobalAlignmentResult {
	return GlobalSequencePairAlign(self.Tokens, other.Tokens)
}

func (self *AnnotatedTextSequence) SimilarityScore(other *AnnotatedTextSequence) float64 {
	return self.AlignWith(other).Score
}

func (self *AnnotatedTextSequence) UnmarshalJSON(b []byte) error {
	trimmedStr := strings.Trim(string(b), "\"")
	*self = *NewAnnotatedTextSequence(trimmedStr)
	return nil
}

func (self *AnnotatedTextSequence) MarshalJSON() ([]byte, error) {
	return json.Marshal(strings.Join(self.Tokens, " "))
}
