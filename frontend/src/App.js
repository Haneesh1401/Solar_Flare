import React, { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
// axios removed; using api layer
import { fetchPredictions, fetchTimeline, fetchModels } from './api/predictions';
import LoadingScreen from './components/LoadingScreen';
import GaugeCard from './components/GaugeCard';
import SolarMap from './components/SolarMap';
import PredictionChart from './components/PredictionChart';
import { dummyPredictions, dummyTimeline } from './data/dummyData';
import Starfield from './components/Starfield';
import SunCanvas from './components/SunCanvas';
import DangerDashboard from './components/DangerDashboard';
import LiveTicker from './components/LiveTicker';
import ExportButtons from './components/ExportButtons';
import Filters from './components/Filters';
import IntensityBadge from './components/IntensityBadge';
import FABCluster from './components/FABCluster';

function App() {
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState(dummyPredictions);
  const [timelineData, setTimelineData] = useState(dummyTimeline);
  const [error, setError] = useState(null);
  const [range, setRange] = useState('24h');
  const [hoverPoint, setHoverPoint] = useState(null);
  const [model, setModel] = useState('default');
  const [models, setModels] = useState([]);

  const fetchData = useCallback(async () => {
    try {
      const [pred, timeline] = await Promise.all([
        fetchPredictions({ range, model }),
        fetchTimeline({ range, model })
      ]);

      setPredictions(pred || dummyPredictions);
      setTimelineData(Array.isArray(timeline) ? timeline : dummyTimeline);
      setError(null);
    } catch (err) {
      console.error('Error fetching data:', err);
      setError('Failed to load data from API. Using dummy data.');
      setPredictions(dummyPredictions);
      setTimelineData(dummyTimeline);
    } finally {
      setLoading(false);
    }
  }, [range, model]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // polling every 30s
  useEffect(() => {
    const id = setInterval(() => {
      fetchData();
    }, 30000);
    return () => clearInterval(id);
  }, [fetchData]);

  useEffect(() => {
    let mounted = true;
    fetchModels().then((list) => {
      if (!mounted) return;
      setModels(list);
      if (list.length && !list.includes(model)) setModel(list[0]);
    });
    return () => { mounted = false; };
  }, [model]);

  if (loading) {
    return <LoadingScreen size="w-32 h-32" />;
  }

  return (
    <div className="min-h-screen bg-cosmic-gradient scanline" id="dashboard-root">
      <Starfield />
      <div className="aurora-layer animate-aurora" aria-hidden="true" />
      <motion.header
        className="bg-space-dark/60 backdrop-blur-md p-6 flex justify-between items-center border-b border-accent-purple/10"
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h1 className="text-2xl font-bold text-accent-cyan neon-title">Solar Flare Dashboard</h1>
        <motion.button
          className="bg-accent-cyan text-space-dark px-4 py-2 rounded-lg font-bold shadow-glow"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={fetchData}
        >
          Refresh Data
        </motion.button>
      </motion.header>

      {error && (
        <div className="p-4 bg-yellow-500 text-white text-center">
          {error}
        </div>
      )}

      <main className="p-6 space-y-6">
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <Filters range={range} setRange={setRange} model={model} setModel={setModel} models={models} onRefresh={fetchData} />
          <ExportButtons targetId="dashboard-root" />
        </div>
        <motion.section
          className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ staggerChildren: 0.2 }}
        >
          <GaugeCard title="C-Class Probability" value={predictions.c} color="#00FFAA" />
          <GaugeCard title="M-Class Probability" value={predictions.m} color="#FFAA00" />
          <GaugeCard title="X-Class Probability" value={predictions.x} color="#FF0000" />
        </motion.section>

        <motion.section
          className="panel p-6 mb-8"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-xl font-bold mb-4 text-text-white neon-title">Solar Disk Map</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <SolarMap />
            <div className="space-y-3">
              <SunCanvas intensity={Math.max(predictions.x, predictions.m)} />
              <IntensityBadge c={predictions.c} m={predictions.m} x={predictions.x} />
            </div>
          </div>
        </motion.section>

        <motion.section
          className="grid grid-cols-1 md:grid-cols-2 gap-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
        >
          <div className="panel p-6">
            <h2 className="text-xl font-bold mb-4 text-text-white neon-title">Timeline</h2>
            <PredictionChart data={timelineData} onPointHover={setHoverPoint} />
            {hoverPoint && (
              <div className="mt-3 text-sm text-text-white/80">
                <span className="font-bold">{hoverPoint.time}:</span>
                <span className="ml-2">C {(hoverPoint.c * 100).toFixed(1)}% • M {(hoverPoint.m * 100).toFixed(1)}% • X {(hoverPoint.x * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
          <DangerDashboard impact={{
            maxClass: predictions.x > 0.3 ? 'X' : predictions.m > 0.4 ? 'M' : 'C',
            sat: predictions.m > 0.4 || predictions.x > 0.2 ? 'High' : 'Medium',
            gps: predictions.m > 0.3 ? 'Medium' : 'Low',
            grid: predictions.x > 0.2 ? 'High' : 'Low',
            aviation: predictions.m > 0.4 ? 'High' : 'Medium',
            comms: predictions.x > 0.15 ? 'Medium' : 'Low',
          }} />
        </motion.section>

        <motion.section
          className="panel p-6 mt-8"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1 }}
        >
          <h2 className="text-xl font-bold mb-4 text-text-white neon-title">ML Model Overview</h2>
          <p className="text-text-white">Accuracy: {Math.round(predictions.accuracy * 100)}% | Confidence: {Math.round(predictions.confidence * 100)}%</p>
        </motion.section>

        <motion.section
          className="panel p-6"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1 }}
        >
          <LiveTicker items={Array.isArray(timelineData) ? timelineData.map(t => ({
            timestamp: t.time || Date.now(),
            class: t.class || (t.x > 0.2 ? 'X' : t.m > 0.2 ? 'M' : 'C'),
            probability: Math.max(t.c || 0, t.m || 0, t.x || 0),
            energy: (t.energy || 1e20).toExponential(2)
          })) : []} />
        </motion.section>
      </main>
      <FABCluster
        onRefresh={fetchData}
        onExportPNG={() => document.querySelector('button[aria-label="Export PNG"]').click()}
        onExportPDF={() => document.querySelector('button[aria-label="Export PDF"]').click()}
      />
    </div>
  );
}

export default App;
