package util

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
)

type JsonDate time.Time

// Implement Marshaler and Unmarshaler interface
func (j *JsonDate) UnmarshalJSON(b []byte) error {
	s := strings.Trim(string(b), "\"")
	t, err := time.Parse("2006-01-02", s)
	if err != nil {
		return err
	}
	*j = JsonDate(t)
	return nil
}

func (j JsonDate) MarshalJSON() ([]byte, error) {
	return json.Marshal(time.Time(j))
}

// Maybe a Format function for printing your date
func (j JsonDate) Format(s string) string {
	t := time.Time(j)
	return t.Format(s)
}

func ReadJsonIntoObject[T any](filePath string) T {
	jsonBytes, err := os.ReadFile(filePath)

	if err != nil {
		fmt.Println(err)
	}

	var jsonData T
	jsonErr := json.Unmarshal(jsonBytes, &jsonData)

	if jsonErr != nil {
		fmt.Println(jsonErr)
	}

	return jsonData

}

func ReadJsonIntoArrayOfObjects[T any](filePath string) []T {

	jsonBytes, err := os.ReadFile(filePath)

	if err != nil {
		fmt.Println(err)
	}

	jsonData := make([]T, 0)
	jsonErr := json.Unmarshal(jsonBytes, &jsonData)

	if jsonErr != nil {
		fmt.Println(jsonErr)
	}

	return jsonData

}
