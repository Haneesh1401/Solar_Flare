import axios from 'axios';

const apiBase = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

export const api = axios.create({
  baseURL: apiBase,
  timeout: 8000,
});

export function getApiBase() {
  return apiBase;
}





