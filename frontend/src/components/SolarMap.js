import React, { useState } from 'react';
import { motion } from 'framer-motion';

const SolarMap = () => {
  const [selectedRegion, setSelectedRegion] = useState(null);

  const regions = [
    { id: 1, x: 50, y: 50, size: 20, activity: 'High', class: 'C' },
    { id: 2, x: 70, y: 30, size: 15, activity: 'Medium', class: 'B' },
    { id: 3, x: 30, y: 70, size: 25, activity: 'Low', class: 'A' },
  ];

  const handleRegionClick = (region) => {
    setSelectedRegion(region);
  };

  return (
    <motion.div
      className="relative w-full h-64 bg-space-dark/60 backdrop-blur-md rounded-lg overflow-hidden border border-accent-purple/20"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      {/* Solar Disk */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-48 h-48 bg-yellow-400/80 rounded-full shadow-neon relative">
          {/* Radar sweep */}
          <motion.div
            className="absolute inset-0 rounded-full"
            style={{
              background: 'conic-gradient(from 0deg, rgba(0,255,255,0.15), transparent 40%)'
            }}
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: 'linear' }}
          />
          {/* Active Regions */}
          {regions.map((region) => (
            <motion.div
              key={region.id}
              className="absolute bg-red-500 rounded-full cursor-pointer"
              style={{ left: `${region.x}%`, top: `${region.y}%`, width: `${region.size}px`, height: `${region.size}px` }}
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => handleRegionClick(region)}
            />
          ))}
        </div>
      </div>

      {/* Tooltip */}
      {selectedRegion && (
        <motion.div
          className="absolute bottom-4 left-1/2 transform -translate-x-1/2 panel p-2 text-text-white text-sm"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          Region {selectedRegion.id}: {selectedRegion.activity} - {selectedRegion.class}-Class
        </motion.div>
      )}
    </motion.div>
  );
};

export default SolarMap;
