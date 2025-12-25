package main

import (
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
	"strconv"

	_ "github.com/lib/pq"

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
	db    *sql.DB
}

func NewEngine(hub *server.Hub, db *sql.DB) *Engine {
	return &Engine{
		books: map[string]*book.Book{},
		hub:   hub,
		db:    db,
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

// persistTrades writes executed trades into Postgres (engine_trades table).
func (e *Engine) persistTrades(trades []book.Trade) {
	if e.db == nil || len(trades) == 0 {
		return
	}

	for _, tr := range trades {
		_, err := e.db.Exec(
			`INSERT INTO engine_trades
				(ts, symbol, price, qty, aggressor_side, maker_order_id, taker_order_id)
			 VALUES ($1, $2, $3, $4, $5, $6, $7)`,
			tr.Ts,
			tr.Symbol,
			tr.Price,
			tr.Qty,
			tr.AggressorSide,
			tr.MakerOrderID,
			tr.TakerOrderID,
		)
		if err != nil {
			log.Printf("[db] insert trade error: %v", err)
		}
	}
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin == "" {
			origin = "http://localhost:5173"
		}

		// allow dev front-end
		if strings.HasPrefix(origin, "http://localhost:5173") {
			w.Header().Set("Access-Control-Allow-Origin", origin)
		}

		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func mustInitDB() *sql.DB {
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		log.Println("[db] no DATABASE_URL set, running without DB")
		return nil
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("[db] open: %v", err)
	}

	if err := db.Ping(); err != nil {
		log.Fatalf("[db] ping: %v", err)
	}

	// Create a simple table just for engine-emitted trades
	_, err = db.Exec(`
        CREATE TABLE IF NOT EXISTS engine_trades (
            id SERIAL PRIMARY KEY,
            ts BIGINT NOT NULL,
            symbol TEXT NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            qty DOUBLE PRECISION NOT NULL,
            aggressor_side TEXT NOT NULL,
            maker_order_id TEXT NOT NULL,
            taker_order_id TEXT NOT NULL
        );
    `)
	if err != nil {
		log.Fatalf("[db] create engine_trades: %v", err)
	}

	log.Println("[db] connected and ensured engine_trades table")
	return db
}

// helper to write trades
func (e *Engine) saveTrades(trades []book.Trade) {
	if e.db == nil || len(trades) == 0 {
		return
	}

	for _, t := range trades {
		_, err := e.db.Exec(
			`INSERT INTO engine_trades
                (ts, symbol, price, qty, aggressor_side, maker_order_id, taker_order_id)
             VALUES ($1, $2, $3, $4, $5, $6, $7)`,
			t.Ts,
			t.Symbol,
			t.Price,
			t.Qty,
			t.AggressorSide,
			t.MakerOrderID,
			t.TakerOrderID,
		)
		if err != nil {
			log.Println("[db] insert trade:", err)
			return
		}
	}
}

func main() {
	port := os.Getenv("ENGINE_PORT")
	if port == "" {
		port = "8080"
	}

	// ---- DB setup --------------------------------------------------------
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("DATABASE_URL not set")
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	if err := db.Ping(); err != nil {
		log.Fatalf("ping db: %v", err)
	}
	log.Println("[db] connected")

	// create engine_trades table if it doesn't exist
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS engine_trades (
			id BIGSERIAL PRIMARY KEY,
			ts BIGINT NOT NULL,
			symbol TEXT NOT NULL,
			price DOUBLE PRECISION NOT NULL,
			qty DOUBLE PRECISION NOT NULL,
			aggressor_side TEXT NOT NULL,
			maker_order_id TEXT,
			taker_order_id TEXT,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
		)
	`)
	if err != nil {
		log.Fatalf("create engine_trades: %v", err)
	}
	log.Println("[db] ensured engine_trades table exists")

	// ---- Engine + WS hub -------------------------------------------------
	hub := server.NewHub()
	eng := NewEngine(hub, db)

	http.HandleFunc("/ws", hub.HandleWS)

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		w.Write([]byte("ok"))
	})

	http.HandleFunc("/order", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req OrderReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		if req.Symbol == "" {
			req.Symbol = "TEST"
		}
		if req.ID == "" {
			http.Error(w, "id required", http.StatusBadRequest)
			return
		}
		if req.Qty <= 0 {
			http.Error(w, "qty must be > 0", http.StatusBadRequest)
			return
		}

		var side book.Side
		switch strings.ToLower(req.Side) {
		case "buy":
			side = book.Buy
		case "sell":
			side = book.Sell
		default:
			http.Error(w, "side must be buy|sell", http.StatusBadRequest)
			return
		}

		var otype book.OrderType
		switch strings.ToLower(req.Type) {
		case "limit":
			if req.Price <= 0 {
				http.Error(w, "price required for limit", http.StatusBadRequest)
				return
			}
			otype = book.Limit
		case "market":
			otype = book.Market
		case "cancel":
			http.Error(w, "cancel not supported in MVP", http.StatusBadRequest)
			return
		default:
			http.Error(w, "type must be limit|market|cancel", http.StatusBadRequest)
			return
		}

		if req.Ts == 0 {
			req.Ts = time.Now().UnixNano()
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

		// persist to DB
		eng.persistTrades(trades)

		// stream to UI
		eng.hub.BroadcastBookUpdate(update)
		if len(trades) > 0 {
			eng.hub.BroadcastTrades(trades)
		}

		resp := OrderResp{BookUpdate: update, Trades: trades}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	http.HandleFunc("/api/trades/recent", func(w http.ResponseWriter, r *http.Request) {
		// we only support GET
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", http.StatusMethodNotAllowed)
			return
		}

		// default limit = 50, allow ?limit=... override
		limit := 50
		if v := r.URL.Query().Get("limit"); v != "" {
			if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 500 {
				limit = n
			}
		}

		type DBTrade struct {
			ID            int64   `json:"id"`
			Ts            int64   `json:"ts"`
			Symbol        string  `json:"symbol"`
			Price         float64 `json:"price"`
			Qty           float64 `json:"qty"`
			AggressorSide string  `json:"aggressorSide"`
			MakerOrderID  string  `json:"makerOrderId"`
			TakerOrderID  string  `json:"takerOrderId"`
		}

		rows, err := db.Query(`
			SELECT id, ts, symbol, price, qty, aggressor_side, maker_order_id, taker_order_id
			  FROM engine_trades
			 ORDER BY id DESC
			 LIMIT $1
		`, limit)
		if err != nil {
			log.Println("[db] recent trades query error:", err)
			http.Error(w, "db error", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var out []DBTrade
		for rows.Next() {
			var t DBTrade
			if err := rows.Scan(
				&t.ID,
				&t.Ts,
				&t.Symbol,
				&t.Price,
				&t.Qty,
				&t.AggressorSide,
				&t.MakerOrderID,
				&t.TakerOrderID,
			); err != nil {
				log.Println("[db] scan trade:", err)
				http.Error(w, "db error", http.StatusInternalServerError)
				return
			}
			out = append(out, t)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(out)
	})

	log.Println("engine listening on :" + port)
	log.Fatal(http.ListenAndServe(":"+port, withCORS(http.DefaultServeMux)))
}
