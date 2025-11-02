// Dummy data for testing the dashboard

export const dummyPredictions = {
  c: 0.25, // C-class probability
  m: 0.15, // M-class
  x: 0.05, // X-class
  confidence: 0.85,
  accuracy: 0.92,
  timestamp: new Date().toISOString(),
};

export const dummyTimeline = [
  { time: 'Now', predicted: 'C', observed: null, risk: 0.25 },
  { time: '24h', predicted: 'M', observed: null, risk: 0.15 },
  { time: '48h', predicted: 'X', observed: null, risk: 0.05 },
  { time: '72h', predicted: 'C', observed: null, risk: 0.20 },
];

export const dummyImpacts = [
  { system: 'Satellite Operations', risk: 'Medium', icon: '🛰️' },
  { system: 'Power Grid', risk: 'Low', icon: '⚡' },
  { system: 'Aviation', risk: 'High', icon: '✈️' },
  { system: 'Communication', risk: 'Medium', icon: '📡' },
];

export const dummyModelInfo = {
  accuracy: 0.92,
  confidence: 0.85,
  features: [
    { name: 'Flux', importance: 0.35 },
    { name: 'Month', importance: 0.20 },
    { name: 'Day', importance: 0.15 },
    { name: 'Hour', importance: 0.30 },
  ],
};

export const dummySolarRegions = [
  { id: 1, x: 100, y: 150, strength: 0.8, sunspots: 5 },
  { id: 2, x: 200, y: 100, strength: 0.6, sunspots: 3 },
  { id: 3, x: 300, y: 200, strength: 0.9, sunspots: 7 },
];
