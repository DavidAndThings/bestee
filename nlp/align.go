package nlp

import "bestee/util"

type GlobalAlignmentResult struct {
	SequenceOne []string
	SequenceTwo []string
	Score       float64
}

func CosineSimilarity(a, b string) float64 {

	tokenDb := GetTokenDatabaseInstance()
	return util.CosineSimilarity(tokenDb.GetEmbedding(a), tokenDb.GetEmbedding(b))

}

func GlobalSequencePairAlign(a, b []string) GlobalAlignmentResult {

	m, n := len(a)+1, len(b)+1
	scoreMat := util.GetMatrixFloat(m, n, 0.0)
	gapPenalty := -1.0

	for i := range m {
		scoreMat[i][0] = gapPenalty * float64(i)
	}

	for j := range n {
		scoreMat[0][j] = gapPenalty * float64(j)
	}

	for i := range m {
		for j := range n {

			if i > 0 && j > 0 {

				match := CosineSimilarity(a[i-1], b[j-1]) + scoreMat[i-1][j-1]
				deletion := scoreMat[i-1][j] + gapPenalty
				insertion := scoreMat[i][j-1] + gapPenalty

				scoreMat[i][j] = util.FindMaxFloat([]float64{match, deletion, insertion})

			}

		}
	}

	return backTraceForAlignment(a, b, gapPenalty, scoreMat)

}

func backTraceForAlignment(
	a, b []string,
	gapPenalty float64,
	scoreMatrix [][]float64,
) GlobalAlignmentResult {

	i, j := len(a), len(b)
	alignmentA, alignmentB := make([]string, 0), make([]string, 0)

	for i > 0 || j > 0 {

		if i > 0 && j > 0 &&
			scoreMatrix[i][j] == CosineSimilarity(a[i-1], b[j-1])+scoreMatrix[i-1][j-1] {

			alignmentA = append([]string{a[i-1]}, alignmentA...)
			alignmentB = append([]string{b[j-1]}, alignmentB...)
			i, j = i-1, j-1

		} else if i > 0 &&
			scoreMatrix[i][j] == scoreMatrix[i-1][j]+gapPenalty {

			alignmentA = append([]string{a[i-1]}, alignmentA...)
			alignmentB = append([]string{"-"}, alignmentB...)
			i = i - 1

		} else if j > 0 &&
			scoreMatrix[i][j] == scoreMatrix[i][j-1]+gapPenalty {

			alignmentA = append([]string{"-"}, alignmentA...)
			alignmentB = append([]string{b[j-1]}, alignmentB...)
			j = j - 1

		}

	}

	return GlobalAlignmentResult{
		SequenceOne: alignmentA,
		SequenceTwo: alignmentB,
		Score:       scoreMatrix[len(a)][len(b)],
	}

}
