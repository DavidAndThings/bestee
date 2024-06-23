package util

import (
	"bufio"
<<<<<<< HEAD
	"crypto/md5"
	"encoding/hex"
=======
>>>>>>> 9377fa9d567d04830c300209a7bf9e4dc7ac28bf
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
<<<<<<< HEAD

func GetMD5Hash(text string) string {

	hasher := md5.New()
	hasher.Write([]byte(text))
	return hex.EncodeToString(hasher.Sum(nil))

}
=======
>>>>>>> 9377fa9d567d04830c300209a7bf9e4dc7ac28bf
