package book

import (
	"container/heap"
	"math"
	"sync"
)

type priceLevel struct {
	price  float64
	orders []*Order // FIFO queue of resting orders at this price
}

type buyHeap []*priceLevel  // max-heap by price
type sellHeap []*priceLevel // min-heap by price

func (h buyHeap) Len() int            { return len(h) }
func (h buyHeap) Less(i, j int) bool  { return h[i].price > h[j].price }
func (h buyHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *buyHeap) Push(x any)         { *h = append(*h, x.(*priceLevel)) }
func (h *buyHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

func (h sellHeap) Len() int           { return len(h) }
func (h sellHeap) Less(i, j int) bool { return h[i].price < h[j].price }
func (h sellHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *sellHeap) Push(x any)        { *h = append(*h, x.(*priceLevel)) }
func (h *sellHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

type Book struct {
	mu sync.Mutex

	symbol string

	buys  buyHeap
	sells sellHeap

	buyMap  map[float64]*priceLevel
	sellMap map[float64]*priceLevel

	lastTrade float64
}

func NewBook(symbol string) *Book {
	b := &Book{
		symbol:  symbol,
		buyMap:  map[float64]*priceLevel{},
		sellMap: map[float64]*priceLevel{},
	}
	heap.Init(&b.buys)
	heap.Init(&b.sells)
	return b
}

func (b *Book) BestBid() (PriceQty, bool) {
	if b.buys.Len() == 0 {
		return PriceQty{}, false
	}
	pl := b.buys[0]
	if len(pl.orders) == 0 {
		return PriceQty{}, false
	}
	sum := 0.0
	for _, o := range pl.orders {
		sum += o.Qty
	}
	return PriceQty{Price: pl.price, Qty: sum}, true
}

func (b *Book) BestAsk() (PriceQty, bool) {
	if b.sells.Len() == 0 {
		return PriceQty{}, false
	}
	pl := b.sells[0]
	if len(pl.orders) == 0 {
		return PriceQty{}, false
	}
	sum := 0.0
	for _, o := range pl.orders {
		sum += o.Qty
	}
	return PriceQty{Price: pl.price, Qty: sum}, true
}

// Add inserts an order and matches it.
// Returns trades and a BookUpdate snapshot.
func (b *Book) Add(o *Order) ([]Trade, BookUpdate) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if o.Ts == 0 {
		o.Ts = NowTs()
	}
	if o.Symbol == "" {
		o.Symbol = b.symbol
	}

	var trades []Trade
	switch o.Type {
	case Market:
		trades = b.matchMarket(o)
	case Limit:
		trades = b.matchLimit(o)
	default:
		// Cancel not supported in MVP
	}

	update := b.snapshot()
	return trades, update
}

func (b *Book) matchLimit(o *Order) (trades []Trade) {
	if o.Side == Buy {
		for o.Qty > 0 && b.sells.Len() > 0 {
			bestAsk := b.sells[0]
			if bestAsk.price > o.Price {
				break // not crossing
			}

			trades = append(trades, b.consumeLevel(o, bestAsk)...)
			if len(bestAsk.orders) == 0 {
				heap.Pop(&b.sells)
				delete(b.sellMap, bestAsk.price)
			}
		}
		if o.Qty > 0 {
			b.enqueue(o)
		}
	} else { // Sell
		for o.Qty > 0 && b.buys.Len() > 0 {
			bestBid := b.buys[0]
			if bestBid.price < o.Price {
				break
			}

			trades = append(trades, b.consumeLevel(o, bestBid)...)
			if len(bestBid.orders) == 0 {
				heap.Pop(&b.buys)
				delete(b.buyMap, bestBid.price)
			}
		}
		if o.Qty > 0 {
			b.enqueue(o)
		}
	}
	return
}

func (b *Book) matchMarket(o *Order) (trades []Trade) {
	if o.Side == Buy {
		for o.Qty > 0 && b.sells.Len() > 0 {
			bestAsk := b.sells[0]

			trades = append(trades, b.consumeLevel(o, bestAsk)...)
			if len(bestAsk.orders) == 0 {
				heap.Pop(&b.sells)
				delete(b.sellMap, bestAsk.price)
			}
		}
	} else {
		for o.Qty > 0 && b.buys.Len() > 0 {
			bestBid := b.buys[0]

			trades = append(trades, b.consumeLevel(o, bestBid)...)
			if len(bestBid.orders) == 0 {
				heap.Pop(&b.buys)
				delete(b.buyMap, bestBid.price)
			}
		}
	}
	return
}

func (b *Book) consumeLevel(aggr *Order, pl *priceLevel) (trades []Trade) {
	for aggr.Qty > 0 && len(pl.orders) > 0 {
		rest := pl.orders[0]

		fill := math.Min(aggr.Qty, rest.Qty)
		aggr.Qty -= fill
		rest.Qty -= fill

		trades = append(trades, Trade{
			Ts:            NowTs(),
			Symbol:        aggr.Symbol,
			Price:         pl.price,
			Qty:           fill,
			AggressorSide: aggr.Side.String(),
			MakerOrderID:  rest.ID,
			TakerOrderID:  aggr.ID,
		})
		b.lastTrade = pl.price

		if rest.Qty == 0 {
			pl.orders = pl.orders[1:]
		}
	}
	return
}

func (b *Book) enqueue(o *Order) {
	var m map[float64]*priceLevel
	var h heap.Interface

	if o.Side == Buy {
		m, h = b.buyMap, &b.buys
	} else {
		m, h = b.sellMap, &b.sells
	}

	pl, ok := m[o.Price]
	if !ok {
		pl = &priceLevel{price: o.Price}
		m[o.Price] = pl
		heap.Push(h, pl)
	}
	pl.orders = append(pl.orders, o)
}

func (b *Book) snapshot() BookUpdate {
	bid, bidOk := b.BestBid()
	ask, askOk := b.BestAsk()

	if !bidOk {
		bid = PriceQty{Price: 0, Qty: 0}
	}
	if !askOk {
		ask = PriceQty{Price: 0, Qty: 0}
	}

	return BookUpdate{
		Ts:             NowTs(),
		Symbol:         b.symbol,
		BestBid:        bid,
		BestAsk:        ask,
		LastTradePrice: b.lastTrade,
	}
}
