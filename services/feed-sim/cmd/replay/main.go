package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
	"io"

	"github.com/segmentio/kafka-go"
)

type Tick struct {
	Ts     int64   `json:"ts"`
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Side   string  `json:"side"` // "buy"|"sell"
	Qty    float64 `json:"qty"`
}

func env(key, def string) string {
	v := os.Getenv(key)
	if v == "" {
		return def
	}
	return v
}

func main() {
	brokers := strings.Split(env("KAFKA_BROKERS", "localhost:9092"), ",")
	topic := env("TICKS_TOPIC", "ticks")
	file := env("DATA_FILE", "../../data/sample/sample_ticks.csv")
	sleepMsStr := env("REPLAY_SLEEP_MS", "0")

	sleepMs, _ := strconv.Atoi(sleepMsStr)

	w := &kafka.Writer{
		Addr:         kafka.TCP(brokers...),
		Topic:        topic,
		Balancer:     &kafka.LeastBytes{},
		RequiredAcks: kafka.RequireAll,
	}
	defer w.Close()

	f, err := os.Open(file)
	if err != nil {
		log.Fatal("open file:", err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = -1

	// header
	header, err := r.Read()
	if err != nil {
		log.Fatal("read header:", err)
	}
	idx := map[string]int{}
	for i, h := range header {
		idx[strings.TrimSpace(h)] = i
	}
	required := []string{"ts", "symbol", "price", "side", "qty"}
	for _, k := range required {
		if _, ok := idx[k]; !ok {
			log.Fatal("missing column:", k)
		}
	}

	ctx := context.Background()
	count := 0

	for {
		row, err := r.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			log.Fatal("read row:", err)
		}

		ts, _ := strconv.ParseInt(row[idx["ts"]], 10, 64)
		price, _ := strconv.ParseFloat(row[idx["price"]], 64)
		qty, _ := strconv.ParseFloat(row[idx["qty"]], 64)
		side := strings.ToLower(strings.TrimSpace(row[idx["side"]]))
		symbol := strings.TrimSpace(row[idx["symbol"]])

		tick := Tick{Ts: ts, Symbol: symbol, Price: price, Side: side, Qty: qty}
		b, _ := json.Marshal(tick)

		err = w.WriteMessages(ctx, kafka.Message{
			Key:   []byte(symbol),
			Value: b,
		})
		if err != nil {
			log.Fatal("write kafka:", err)
		}

		count++
		if sleepMs > 0 {
			time.Sleep(time.Duration(sleepMs) * time.Millisecond)
		}
	}

	fmt.Printf("Replayed %d ticks to topic %s\n", count, topic)
}
