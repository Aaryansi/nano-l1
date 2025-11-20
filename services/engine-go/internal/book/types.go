package book

import "time"

type Side int

const (
	Buy Side = iota
	Sell
)

func (s Side) String() string {
	if s == Buy {
		return "buy"
	}
	return "sell"
}

type OrderType int

const (
	Limit OrderType = iota
	Market
	Cancel // reserved for later
)

type Order struct {
	ID        string
	Ts        int64
	Symbol    string
	Side      Side
	Type      OrderType
	Price     float64 // only for limit
	Qty       float64
}

type Trade struct {
	Ts             int64   `json:"ts"`
	Symbol         string  `json:"symbol"`
	Price          float64 `json:"price"`
	Qty            float64 `json:"qty"`
	AggressorSide  string  `json:"aggressorSide"`
	MakerOrderID   string  `json:"makerOrderId,omitempty"`
	TakerOrderID   string  `json:"takerOrderId,omitempty"`
}

type PriceQty struct {
	Price float64 `json:"price"`
	Qty   float64 `json:"qty"`
}

// Matches shared/schemas/jsonschema/book_update.schema.json
type BookUpdate struct {
	Ts             int64    `json:"ts"`
	Symbol         string   `json:"symbol"`
	BestBid        PriceQty `json:"bestBid"`
	BestAsk        PriceQty `json:"bestAsk"`
	LastTradePrice float64  `json:"lastTradePrice,omitempty"`
}

func NowTs() int64 { return time.Now().UnixNano() }
