import { apiBase, getToken } from "./api";

export async function openFeedSocket(onMessage, onError) {
  const token = await getToken();
  const url = apiBase().replace(/^http/, "ws") + `/transactions/feed?token=${encodeURIComponent(token)}`;
  const ws = new WebSocket(url);
  ws.onopen = () => {
    ws.send("subscribe");
  };
  ws.onmessage = (evt) => {
    try {
      onMessage(JSON.parse(evt.data));
    } catch (_) {
      onMessage(evt.data);
    }
  };
  ws.onerror = (err) => {
    if (onError) {
      onError(err);
    }
  };
  return ws;
}

