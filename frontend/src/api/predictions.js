import { api } from './client';

// Use realtime endpoint to get latest prediction mapped to UI shape
export async function fetchPredictions({ range = '24h', model } = {}) {
  // Optional: switch model via /models/switch
  if (model && model !== 'default') {
    try { await api.post('/models/switch', { model_name: model }); } catch {}
  }
  const res = await api.get('/predict/realtime');
  const pred = res.data?.prediction;
  const probs = pred?.probabilities || {};
  return {
    c: probs['C'] || 0,
    m: probs['M'] || 0,
    x: probs['X'] || 0,
    accuracy: pred?.confidence || 0.9,
    confidence: pred?.confidence || 0.9
  };
}

// No native timeline endpoint; return empty and let UI handle gracefully
export async function fetchTimeline({ range = '24h', model } = {}) {
  return [];
}

export async function fetchModels() {
  try {
    const res = await api.get('/models');
    const arr = Array.isArray(res.data) ? res.data : [];
    const names = [...new Set(arr.map((m) => m.name))];
    return names;
  } catch {
    return [];
  }
}


