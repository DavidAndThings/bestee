package util

import (
	"bufio"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
)

func ReadJsonIntoMap(filePath string) map[string]interface{} {
	jsonBytes, err := os.ReadFile(filePath)

	if err != nil {
		fmt.Println(err)
	}

	var jsonData map[string]interface{} = make(map[string]interface{})
	jsonErr := json.Unmarshal(jsonBytes, &jsonData)

	if jsonErr != nil {
		fmt.Println(jsonErr)
	}

	return jsonData

}

func ReadJsonIntoArray(filePath string) []map[string]interface{} {

	jsonBytes, err := os.ReadFile(filePath)

	if err != nil {
		fmt.Println(err)
	}

	var jsonData []map[string]interface{} = make([]map[string]interface{}, 0)
	jsonErr := json.Unmarshal(jsonBytes, &jsonData)

	if jsonErr != nil {
		fmt.Println(jsonErr)
	}

	return jsonData

}

func GetExcutableDir() string {

	ex, err := os.Executable()
	if err != nil {
		panic(err)
	}

	return filepath.Dir(ex)

}

func ReadConfigJson() map[string]interface{} {

	exeDir := GetExcutableDir()
	return ReadJsonIntoMap(exeDir + "/config.json")

}

func ReadIntoStrArray(path string) ([]string, error) {

	file, err := os.Open(path)

	if err != nil {
		return nil, err
	}

	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	return lines, scanner.Err()

}

func GetMD5Hash(text string) string {

	hasher := md5.New()
	hasher.Write([]byte(text))
	return hex.EncodeToString(hasher.Sum(nil))

}

func StrArrayToFloatArray(input []string) []float64 {

	ans := make([]float64, len(input))

	for i, val := range input {

		newVal, err := strconv.ParseFloat(val, 64)

		if err != nil {
			panic(fmt.Sprintf("Cannot convert str: %s to float!", val))
		}

		ans[i] = newVal

	}

	return ans

}

func CosineSimilarity(a []float64, b []float64) float64 {
	return DotProduct(a, b) / (Magnitude(a) * Magnitude(b))
}

func DotProduct(a []float64, b []float64) float64 {

	if len(a) != len(b) {

		panic(
			fmt.Sprintf(
				"Input vectors: %s and %s are not the same length!",
				fmt.Sprint(a),
				fmt.Sprint(b),
			),
		)

	}

	ans := 0.0

	for i := range len(a) {
		ans += a[i] * b[i]
	}

	return ans

}

func Magnitude(a []float64) float64 {

	ans := 0.0

	for _, val := range a {
		ans += val * val
	}

	return math.Sqrt(ans)

}
