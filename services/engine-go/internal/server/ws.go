package server

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"

	"nano-l1/engine-go/internal/book"

	"github.com/gorilla/websocket"
)

type Hub struct {
	mu      sync.Mutex
	clients map[*websocket.Conn]bool
}

func NewHub() *Hub {
	return &Hub{clients: map[*websocket.Conn]bool{}}
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func (h *Hub) HandleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("ws upgrade:", err)
		return
	}

	h.mu.Lock()
	h.clients[conn] = true
	h.mu.Unlock()

	conn.SetCloseHandler(func(code int, text string) error {
		h.mu.Lock()
		delete(h.clients, conn)
		h.mu.Unlock()
		return nil
	})
}

// Generic event envelope for WS (and later Kafka if we want)
type Envelope struct {
	EventType string      `json:"eventType"`
	Data      interface{} `json:"data"`
}

func (h *Hub) broadcastEnvelope(env Envelope) {
	data, _ := json.Marshal(env)

	h.mu.Lock()
	defer h.mu.Unlock()

	for c := range h.clients {
		if err := c.WriteMessage(websocket.TextMessage, data); err != nil {
			c.Close()
			delete(h.clients, c)
		}
	}
}

// Send top-of-book snapshot/state
func (h *Hub) BroadcastBookUpdate(update book.BookUpdate) {
	h.broadcastEnvelope(Envelope{
		EventType: "book_update",
		Data:      update,
	})
}

// Send trades as a batch event
func (h *Hub) BroadcastTrades(trades []book.Trade) {
	if trades == nil {
		trades = []book.Trade{}
	}
	h.broadcastEnvelope(Envelope{
		EventType: "trades",
		Data:      trades,
	})
}
