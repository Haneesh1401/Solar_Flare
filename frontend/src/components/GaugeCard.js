import React from 'react';
import { motion } from 'framer-motion';

const GaugeCard = ({ title, value, max = 1, color = '#00FFFF' }) => {
  const percentage = (value / max) * 100;

  return (
    <motion.div
      className="panel p-6 shadow-neon flex flex-col items-center hover:shadow-[0_0_24px_rgba(0,255,255,0.25)] transition-shadow"
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h3 className="text-xl font-bold mb-4 text-text-white neon-title tracking-widest uppercase">{title}</h3>
      <div className="relative w-32 h-32">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 36 36">
          <path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="#374151"
            strokeWidth="3"
            strokeLinecap="round"
          />
          <motion.path
            d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke={color}
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={`${percentage * 0.1}, 100`}
            initial={{ strokeDasharray: '0, 100' }}
            animate={{ strokeDasharray: `${percentage * 0.1}, 100` }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <motion.span
            className="text-2xl font-bold text-accent-cyan font-exo"
            initial={{ opacity: 0.6 }}
            animate={{ opacity: 1 }}
            transition={{ repeat: Infinity, repeatType: 'reverse', duration: 1.2 }}
          >
            {(percentage).toFixed(1)}%
          </motion.span>
        </div>
      </div>
    </motion.div>
  );
};

export default GaugeCard;
