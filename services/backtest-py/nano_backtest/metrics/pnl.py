def update_pnl(pos, cash, trades):
    """
    Very naive PnL:
    - If aggressorSide == buy => we bought (pos += qty, cash -= price*qty)
    - If sell => we sold (pos -= qty, cash += price*qty)
    """
    for t in trades:
        qty = float(t["qty"])
        price = float(t["price"])
        side = t["aggressorSide"]
        if side == "buy":
            pos += qty
            cash -= price * qty
        else:
            pos -= qty
            cash += price * qty
    return pos, cash

def mark_to_market(pos, cash, last_price):
    return cash + pos * last_price
