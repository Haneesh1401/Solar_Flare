import React from 'react';
import { motion } from 'framer-motion';

function FABCluster({ onRefresh, onExportPNG, onExportPDF }) {
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3">
      <motion.button
        className="px-4 py-3 rounded-full bg-accent-cyan text-space-dark font-bold shadow-neon"
        whileHover={{ scale: 1.06, boxShadow: '0 0 20px rgba(0,255,255,0.6)' }}
        whileTap={{ scale: 0.96 }}
        onClick={onRefresh}
        aria-label="Refresh data"
      >
        ⟳ Refresh
      </motion.button>
      <div className="flex gap-2">
        <motion.button
          className="px-3 py-2 rounded-full bg-accent-orange text-space-dark font-bold"
          whileHover={{ scale: 1.06 }}
          whileTap={{ scale: 0.96 }}
          onClick={onExportPNG}
          aria-label="Export PNG"
        >PNG</motion.button>
        <motion.button
          className="px-3 py-2 rounded-full bg-accent-magenta text-space-dark font-bold"
          whileHover={{ scale: 1.06 }}
          whileTap={{ scale: 0.96 }}
          onClick={onExportPDF}
          aria-label="Export PDF"
        >PDF</motion.button>
      </div>
    </div>
  );
}

export default FABCluster;





