import React from 'react';

function LiveTicker({ items = [] }) {
  return (
    <div className="panel overflow-hidden" aria-live="polite" aria-atomic="true">
      <div className="flex items-center gap-3 mb-2">
        <div className="w-2 h-2 rounded-full bg-accent-cyan animate-pulseGlow" aria-hidden />
        <h2 className="text-xl font-bold neon-title">Latest Predictions</h2>
      </div>
      <ul className="space-y-1">
        {items.slice(0, 6).map((it, idx) => (
          <li key={idx} className="flex justify-between text-sm border-b border-accent-purple/10 py-1">
            <span className="text-text-white/80">{new Date(it.timestamp).toLocaleString()}</span>
            <span className="font-exo">
              {it.class} • {(it.probability * 100).toFixed(1)}% • {it.energy} J
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default LiveTicker;


