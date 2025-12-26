package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/lib/pq"

	"nano-l1/engine-go/internal/book"
	"nano-l1/engine-go/internal/server"
	"nano-l1/engine-go/internal/stream"
)

// ---------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------

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

type StatsResp struct {
	TradesInDB int64 `json:"tradesInDB"`
}

// DB-backed PnL response
type PnLResp struct {
	Symbol      string  `json:"symbol"`
	TradeCount  int64   `json:"tradeCount"`
	Position    float64 `json:"position"`
	AvgEntry    float64 `json:"avgEntryPrice"`
	LastPrice   float64 `json:"lastPrice"`
	RealizedPnL float64 `json:"realizedPnL"`
	Unrealized  float64 `json:"unrealizedPnL"`
	TotalPnL    float64 `json:"totalPnL"`
}

// ---------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------

type Engine struct {
	mu    sync.Mutex
	books map[string]*book.Book
	hub   *server.Hub
	db    *sql.DB
	bus   *stream.Bus // Kafka bus (optional)
}

func NewEngine(hub *server.Hub, db *sql.DB, bus *stream.Bus) *Engine {
	return &Engine{
		books: map[string]*book.Book{},
		hub:   hub,
		db:    db,
		bus:   bus,
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

// ---------------------------------------------------------------------
// CORS wrapper
// ---------------------------------------------------------------------

func withCORS(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        origin := r.Header.Get("Origin")

        // For dev, just reflect whatever origin hit us (if any).
        if origin != "" {
            w.Header().Set("Access-Control-Allow-Origin", origin)
        } else {
            // Fallback – useful for curl, etc.
            w.Header().Set("Access-Control-Allow-Origin", "*")
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


// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------

func main() {
	port := os.Getenv("ENGINE_PORT")
	if port == "" {
		port = "8080"
	}

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

	// Ensure engine_trades table
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

	// -----------------------------------------------------------------
	// Kafka bus init (feed-sim ↔ engine ↔ trades topic)
	// -----------------------------------------------------------------

	var bus *stream.Bus

	brokersCSV := os.Getenv("KAFKA_BROKERS")
	if brokersCSV == "" {
		// default docker-compose host
		brokersCSV = "kafka:9092"
	}
	bus = stream.NewBus(brokersCSV)
	if bus != nil {
		log.Printf("[kafka] bus initialised with brokers: %s", brokersCSV)
	}

	// context for background Kafka consumers
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Consume ticks from Kafka (feed-sim → Kafka → engine)
	if bus != nil {
		go bus.ConsumeTicks(ctx, func(t stream.Tick) {
			log.Printf("[kafka] tick: %+v", t)
			// later: you can push ticks into a symbol book or strategy engine here
		})
	}

	hub := server.NewHub()
	eng := NewEngine(hub, db, bus)

	// -----------------------------------------------------------------
	// WS + health
	// -----------------------------------------------------------------

	http.HandleFunc("/ws", hub.HandleWS)

	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	// -----------------------------------------------------------------
	// /stats – simple count of trades
	// -----------------------------------------------------------------

	http.HandleFunc("/stats", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if eng.db == nil {
			_ = json.NewEncoder(w).Encode(StatsResp{TradesInDB: 0})
			return
		}

		var count int64
		err := eng.db.QueryRow(`SELECT COUNT(*) FROM engine_trades`).Scan(&count)
		if err != nil {
			log.Printf("[db] stats error: %v", err)
			_ = json.NewEncoder(w).Encode(StatsResp{TradesInDB: 0})
			return
		}

		_ = json.NewEncoder(w).Encode(StatsResp{TradesInDB: count})
	})

	// -----------------------------------------------------------------
	// /order – manual order entry
	// -----------------------------------------------------------------

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

		// publish to Kafka trades topic (fire-and-forget)
		if eng.bus != nil && len(trades) > 0 {
			go eng.bus.PublishTrades(context.Background(), trades)
		}

		resp := OrderResp{BookUpdate: update, Trades: trades}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	// -----------------------------------------------------------------
	// /api/trades/recent – last N trades from engine_trades
	// -----------------------------------------------------------------

	http.HandleFunc("/api/trades/recent", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", http.StatusMethodNotAllowed)
			return
		}

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
			AggressorSide string  `json:"aggressor_side"`
			MakerOrderID  string  `json:"maker_order_id"`
			TakerOrderID  string  `json:"taker_order_id"`
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
		_ = json.NewEncoder(w).Encode(map[string]any{
			"trades": out,
		})
	})

	// -----------------------------------------------------------------
	// /pnl – simple P&L aggregation from engine_trades
	// -----------------------------------------------------------------

	http.HandleFunc("/pnl", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "GET only", http.StatusMethodNotAllowed)
			return
		}

		symbol := r.URL.Query().Get("symbol")
		if symbol == "" {
			symbol = "TEST"
		}

		rows, err := db.Query(`
			SELECT price, qty, aggressor_side
			  FROM engine_trades
			 WHERE symbol = $1
			 ORDER BY id ASC
		`, symbol)
		if err != nil {
			log.Println("[db] pnl query error:", err)
			http.Error(w, "db error", http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var (
			buyQty, buyNotional   float64
			sellQty, sellNotional float64
			lastPrice             float64
			count                 int64
		)

		for rows.Next() {
			var p, q float64
			var side string
			if err := rows.Scan(&p, &q, &side); err != nil {
				log.Println("[db] scan pnl row:", err)
				http.Error(w, "db error", http.StatusInternalServerError)
				return
			}
			side = strings.ToLower(side)

			if side == "buy" {
				buyQty += q
				buyNotional += p * q
			} else if side == "sell" {
				sellQty += q
				sellNotional += p * q
			}
			lastPrice = p
			count++
		}

		position := buyQty - sellQty
		totalPnL := sellNotional - buyNotional

		var avgEntry float64
		if buyQty > 0 {
			avgEntry = buyNotional / buyQty
		}

		var unrealized float64
		if position != 0 && lastPrice != 0 && avgEntry != 0 {
			unrealized = position * (lastPrice - avgEntry)
		}
		realized := totalPnL - unrealized

		resp := PnLResp{
			Symbol:      symbol,
			TradeCount:  count,
			Position:    position,
			AvgEntry:    avgEntry,
			LastPrice:   lastPrice,
			RealizedPnL: realized,
			Unrealized:  unrealized,
			TotalPnL:    totalPnL,
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	log.Println("engine listening on :" + port)
	log.Fatal(http.ListenAndServe(":"+port, withCORS(http.DefaultServeMux)))
}
