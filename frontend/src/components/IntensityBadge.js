import React from 'react';

function getBadgeStyle(cls) {
  switch (cls) {
    case 'X':
      return 'bg-red-500/20 text-red-300 border border-red-400/30 shadow-neon';
    case 'M':
      return 'bg-orange-500/20 text-orange-300 border border-orange-400/30';
    case 'C':
      return 'bg-yellow-500/20 text-yellow-200 border border-yellow-400/30';
    default:
      return 'bg-cyan-500/20 text-cyan-200 border border-cyan-400/30';
  }
}

function IntensityBadge({ c = 0, m = 0, x = 0 }) {
  const maxVal = Math.max(c || 0, m || 0, x || 0);
  const cls = maxVal === x ? 'X' : maxVal === m ? 'M' : 'C';
  const percent = (maxVal * 100).toFixed(1);
  return (
    <div className={`inline-flex items-center gap-2 px-3 py-1 rounded ${getBadgeStyle(cls)}`} role="status" aria-live="polite">
      <span className="font-bold">Intensity:</span>
      <span className="font-exo text-lg">{cls}</span>
      <span className="text-text-white/70">{percent}%</span>
    </div>
  );
}

export default IntensityBadge;



