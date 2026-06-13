import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
let tokenCache = null;

export async function getToken() {
  if (tokenCache) {
    return tokenCache;
  }
  const fromStorage = localStorage.getItem("kavach_token");
  if (fromStorage) {
    tokenCache = fromStorage;
    return tokenCache;
  }
  const health = await axios.get(`${API_BASE}/health`);
  const token = health?.data?.demo_token;
  if (token) {
    tokenCache = token;
    localStorage.setItem("kavach_token", token);
  }
  return tokenCache;
}

export async function apiGet(path) {
  const token = await getToken();
  return axios.get(`${API_BASE}${path}`, {
    headers: { Authorization: `Bearer ${token}` }
  });
}

export async function apiPost(path, data) {
  const token = await getToken();
  return axios.post(`${API_BASE}${path}`, data, {
    headers: { Authorization: `Bearer ${token}` }
  });
}

export function apiBase() {
  return API_BASE;
}

