// src/lib/wsClient.js

/**
 * Connect to the engine WebSocket with simple reconnect + status callbacks.
 *
 * @param {string} url
 * @param {(msg: any) => void} onMessage
 * @param {(status: 'connecting'|'open'|'reconnecting'|'closed'|'error') => void} [onStatus]
 * @returns {{ close: () => void }}
 */
export function connectWS(url, onMessage, onStatus) {
  let ws;
  let closedByUser = false;

  const setStatus = (s) => {
    if (onStatus) onStatus(s);
  };

  const connect = () => {
    setStatus('connecting');

    ws = new WebSocket(url);

    ws.onopen = () => {
      setStatus('open');
      // console.log('[ws] open', url);
    };

    ws.onerror = (err) => {
      console.error('[ws] error', err);
      setStatus('error');
    };

    ws.onclose = () => {
      if (closedByUser) {
        setStatus('closed');
        return;
      }

      // auto-reconnect with small backoff
      setStatus('reconnecting');
      setTimeout(connect, 1000);
    };

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        onMessage && onMessage(msg);
      } catch (e) {
        console.error('[ws] bad message', e, evt.data);
      }
    };
  };

  connect();

  return {
    close() {
      closedByUser = true;
      try {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      } catch (e) {
        console.error('[ws] close error', e);
      }
    },
  };
}
