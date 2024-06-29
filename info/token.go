package info

import (
	"bestee/util"
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
)

var tokenDbInstance *TokenDatabase

type TokenDatabase struct {
	redis    *redis.Client
	redisCtx context.Context
}

func GetTokenDatabaseInstance() *TokenDatabase {

	if tokenDbInstance == nil {

		config := util.ReadConfigJson()["token_redis"].(map[string]interface{})

		tokenDbInstance = &TokenDatabase{
			redis: redis.NewClient(&redis.Options{
				Addr:     fmt.Sprintf("%s:%d", config["host"].(string), int(config["port"].(float64))),
				Username: config["username"].(string),
				Password: config["password"].(string),
				DB:       int(config["db"].(float64)),
			}),
			redisCtx: context.Background(),
		}
	}

	return tokenDbInstance

}

func (db *TokenDatabase) HasToken(query string) int {

	ans, err := db.redis.Exists(db.redisCtx, query).Result()

	if err != nil {
		return 0
	}

	return int(ans)

}

func (db *TokenDatabase) GetEmbedding(query string) []float64 {

	vec, err := db.redis.LRange(db.redisCtx, query, 0, -1).Result()

	if err != nil {
		return make([]float64, 300)
	}

	return util.StrArrayToFloatArray(vec)

}
