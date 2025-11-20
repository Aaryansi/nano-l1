export function connectWS(url, onEvent) {
  const ws = new WebSocket(url);

  ws.onopen = () => console.log("[WS] connected", url);
  ws.onclose = () => console.log("[WS] closed");
  ws.onerror = (e) => console.log("[WS] error", e);

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      onEvent(msg);
    } catch (err) {
      console.error("[WS] bad message", e.data, err);
    }
  };

  return ws;
}
