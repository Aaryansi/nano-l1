package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"sync"

	"nano-l1/engine-go/internal/book"
	"nano-l1/engine-go/internal/server"
)

type OrderReq struct {
	ID     string  `json:"id"`
	Ts     int64   `json:"ts"`
	Symbol string  `json:"symbol"`
	Side   string  `json:"side"` // buy|sell
	Type   string  `json:"type"` // limit|market|cancel
	Price  float64 `json:"price"`
	Qty    float64 `json:"qty"`
}

type OrderResp struct {
	BookUpdate book.BookUpdate `json:"bookUpdate"`
	Trades     []book.Trade    `json:"trades"`
}

type Engine struct {
	mu    sync.Mutex
	books map[string]*book.Book
	hub   *server.Hub
}

func NewEngine(hub *server.Hub) *Engine {
	return &Engine{
		books: map[string]*book.Book{},
		hub:   hub,
	}
}

func (e *Engine) getBook(symbol string) *book.Book {
	e.mu.Lock()
	defer e.mu.Unlock()

	bk, ok := e.books[symbol]
	if !ok {
		bk = book.NewBook(symbol)
		e.books[symbol] = bk
	}
	return bk
}

func main() {
	port := os.Getenv("ENGINE_PORT")
	if port == "" {
		port = "8080"
	}

	hub := server.NewHub()
	eng := NewEngine(hub)

	http.HandleFunc("/ws", hub.HandleWS)

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte("ok"))
	})

	http.HandleFunc("/order", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}

		var req OrderReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}

		if req.Symbol == "" {
			req.Symbol = "TEST"
		}
		if req.ID == "" {
			http.Error(w, "id required", 400)
			return
		}
		if req.Qty <= 0 {
			http.Error(w, "qty must be > 0", 400)
			return
		}

		var side book.Side
		switch req.Side {
		case "buy":
			side = book.Buy
		case "sell":
			side = book.Sell
		default:
			http.Error(w, "side must be buy|sell", 400)
			return
		}

		var otype book.OrderType
		switch req.Type {
		case "limit":
			if req.Price <= 0 {
				http.Error(w, "price required for limit", 400)
				return
			}
			otype = book.Limit
		case "market":
			otype = book.Market
		case "cancel":
			http.Error(w, "cancel not supported in MVP", 400)
			return
		default:
			http.Error(w, "type must be limit|market|cancel", 400)
			return
		}

		ord := &book.Order{
			ID:     req.ID,
			Ts:     req.Ts,
			Symbol: req.Symbol,
			Side:   side,
			Type:   otype,
			Price:  req.Price,
			Qty:    req.Qty,
		}

		bk := eng.getBook(req.Symbol)
		trades, update := bk.Add(ord)
		if trades == nil {
			trades = []book.Trade{}
		}

		// stream to UI as events
		eng.hub.BroadcastBookUpdate(update)
		if len(trades) > 0 {
			eng.hub.BroadcastTrades(trades)
		}

		resp := OrderResp{BookUpdate: update, Trades: trades}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	log.Println("engine listening on :" + port)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}
