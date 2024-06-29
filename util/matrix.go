package util

func GetMatrixFloat(m, n int, initialValue float64) [][]float64 {

	ans := make([][]float64, m)

	for i := range m {
		ans[i] = make([]float64, n)
	}

	for i := range m {
		for j := range n {
			ans[i][j] = initialValue
		}
	}

	return ans

}

func GetMatrixInt(m, n, initialValue int) [][]int {

	ans := make([][]int, m)

	for i := range m {
		ans[i] = make([]int, n)
	}

	for i := range m {
		for j := range n {
			ans[i][j] = initialValue
		}
	}

	return ans

}
