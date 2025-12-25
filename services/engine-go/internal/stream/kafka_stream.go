package stream

import (
	"context"
	"encoding/json"
	"log"
	"strings"
	"time"

	"nano-l1/engine-go/internal/book"

	"github.com/segmentio/kafka-go"
)

type Tick struct {
	Ts     int64   `json:"ts"`
	Symbol string  `json:"symbol"`
	Price  float64 `json:"price"`
	Side   string  `json:"side"`
	Qty    float64 `json:"qty"`
}

type Bus struct {
	Brokers []string

	TicksTopic       string
	BookUpdatesTopic string
	TradesTopic      string

	bookW   *kafka.Writer
	tradesW *kafka.Writer
}

func NewBus(brokersCSV string) *Bus {
	brokers := []string{}
	for _, b := range strings.Split(brokersCSV, ",") {
		b = strings.TrimSpace(b)
		if b != "" {
			brokers = append(brokers, b)
		}
	}
	if len(brokers) == 0 {
		return nil
	}

	bus := &Bus{
		Brokers:          brokers,
		TicksTopic:       "ticks",
		BookUpdatesTopic: "book_updates",
		TradesTopic:      "trades",
		bookW: &kafka.Writer{
			Addr:         kafka.TCP(brokers...),
			Topic:        "book_updates",
			Balancer:     &kafka.LeastBytes{},
			RequiredAcks: kafka.RequireAll,
		},
		tradesW: &kafka.Writer{
			Addr:         kafka.TCP(brokers...),
			Topic:        "trades",
			Balancer:     &kafka.LeastBytes{},
			RequiredAcks: kafka.RequireAll,
		},
	}
	return bus
}

func (b *Bus) Close() {
	if b == nil {
		return
	}
	_ = b.bookW.Close()
	_ = b.tradesW.Close()
}

func (b *Bus) PublishBookUpdate(ctx context.Context, u book.BookUpdate) {
	if b == nil {
		return
	}
	msg, _ := json.Marshal(u)
	_ = b.bookW.WriteMessages(ctx, kafka.Message{
		Key:   []byte(u.Symbol),
		Value: msg,
		Time:  time.Now(),
	})
}

func (b *Bus) PublishTrades(ctx context.Context, trades []book.Trade) {
	if b == nil || len(trades) == 0 {
		return
	}
	for _, t := range trades {
		msg, _ := json.Marshal(t)
		_ = b.tradesW.WriteMessages(ctx, kafka.Message{
			Key:   []byte(t.Symbol),
			Value: msg,
			Time:  time.Now(),
		})
	}
}

// Consume ticks and call onTick for each tick.
// This runs forever until ctx is canceled.
func (b *Bus) ConsumeTicks(ctx context.Context, onTick func(Tick)) {
	if b == nil {
		return
	}

	r := kafka.NewReader(kafka.ReaderConfig{
		Brokers: b.Brokers,
		Topic:   b.TicksTopic,
		GroupID: "engine-ticks",
		MinBytes: 1,
		MaxBytes: 10e6,
	})
	defer r.Close()

	log.Println("[kafka] consuming ticks from", b.TicksTopic)

	for {
		m, err := r.ReadMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			log.Println("[kafka] read tick error:", err)
			continue
		}

		var t Tick
		if err := json.Unmarshal(m.Value, &t); err != nil {
			log.Println("[kafka] bad tick:", err)
			continue
		}

		onTick(t)
	}
}
