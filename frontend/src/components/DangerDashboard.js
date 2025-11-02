import React from 'react';
import { motion } from 'framer-motion';

const risks = [
  { label: 'Satellites', icon: '🛰️', key: 'sat', color: 'text-yellow-400' },
  { label: 'GPS/Navigation', icon: '📡', key: 'gps', color: 'text-cyan-300' },
  { label: 'Power Grids', icon: '⚡', key: 'grid', color: 'text-orange-400' },
  { label: 'Aviation', icon: '✈️', key: 'aviation', color: 'text-pink-400' },
  { label: 'Comms', icon: '📶', key: 'comms', color: 'text-purple-400' },
];

function RiskBadge({ level = 'Low' }) {
  const map = {
    Low: 'bg-green-500/20 text-green-300 border border-green-400/30',
    Medium: 'bg-yellow-500/20 text-yellow-300 border border-yellow-400/30',
    High: 'bg-red-500/20 text-red-300 border border-red-400/30 animate-pulse',
  };
  return <span className={`px-2 py-0.5 rounded ${map[level]}`}>{level}</span>;
}

function DangerDashboard({ impact = {} }) {
  return (
    <div className="panel">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold neon-title">Danger Dashboard</h2>
        {['M', 'X'].includes(impact.maxClass) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="px-3 py-1 rounded border border-red-400/40 text-red-300 shadow-neon">
            High Flare Alert: {impact.maxClass}
          </motion.div>
        )}
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {risks.map((r) => (
          <div key={r.key} className="flex items-center justify-between bg-space-dark/40 rounded-md p-3 border border-accent-purple/10">
            <div className="flex items-center gap-3">
              <span className={`text-2xl ${r.color}`} aria-hidden>
                {r.icon}
              </span>
              <span>{r.label}</span>
            </div>
            <RiskBadge level={impact[r.key] || 'Low'} />
          </div>
        ))}
      </div>
    </div>
  );
}

export default DangerDashboard;


