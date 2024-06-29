package util

func FindMinFloat(array []float64) float64 {

	if len(array) == 0 {
		panic("Minimum undefined for empty array!")
	}

	ans := array[0]

	for _, i := range array {
		if i < ans {
			ans = i
		}
	}

	return ans

}

func FindMaxFloat(array []float64) float64 {

	if len(array) == 0 {
		panic("Maximum undefined for empty array!")
	}

	ans := array[0]

	for _, i := range array {
		if i > ans {
			ans = i
		}
	}

	return ans

}
