import React from 'react';
import { motion } from 'framer-motion';

const LoadingScreen = ({ size = 'w-16 h-16' }) => {
  return (
    <div className="flex items-center justify-center min-h-screen bg-space-dark">
      <motion.div
        className={`border-4 border-accent-cyan border-t-transparent rounded-full ${size}`}
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
      />
      <motion.p
        className="ml-4 text-accent-cyan text-xl"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, repeat: Infinity, repeatType: 'reverse' }}
      >
        Loading Solar Data...
      </motion.p>
    </div>
  );
};

export default LoadingScreen;
