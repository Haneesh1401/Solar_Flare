import React from 'react';

function Filters({ range, setRange, model, setModel, models = [], onRefresh }) {
  return (
    <div className="panel flex flex-wrap items-center gap-3" role="group" aria-label="Filters">
      <label className="flex items-center gap-2">
        <span>Date Range</span>
        <select
          className="bg-space-dark/60 border border-accent-purple/20 rounded px-2 py-1"
          value={range}
          onChange={(e) => setRange(e.target.value)}
          aria-label="Select date range"
        >
          <option value="24h">Last 24h</option>
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="1y">Last year</option>
        </select>
      </label>
      <label className="flex items-center gap-2">
        <span>Model</span>
        <select
          className="bg-space-dark/60 border border-accent-purple/20 rounded px-2 py-1"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          aria-label="Select ML model"
        >
          {models.length === 0 && <option value="default">Default</option>}
          {models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </label>
      <button className="px-3 py-1 rounded bg-accent-magenta text-space-dark font-bold" onClick={onRefresh} aria-label="Refresh data">
        Refresh
      </button>
    </div>
  );
}

export default Filters;


