package util

import (
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
