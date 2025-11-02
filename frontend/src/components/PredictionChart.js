import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const PredictionChart = ({ data, onPointHover }) => {
  const chartData = {
    labels: data.map(item => item.time),
    datasets: [
      {
        label: 'C-Class Probability',
        data: data.map(item => item.c || 0),
        borderColor: '#00FFAA',
        backgroundColor: 'rgba(0, 255, 170, 0.2)',
      },
      {
        label: 'M-Class Probability',
        data: data.map(item => item.m || 0),
        borderColor: '#FFAA00',
        backgroundColor: 'rgba(255, 170, 0, 0.2)',
      },
      {
        label: 'X-Class Probability',
        data: data.map(item => item.x || 0),
        borderColor: '#FF0000',
        backgroundColor: 'rgba(255, 0, 0, 0.2)',
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`,
          afterBody: () => 'Click a point for details',
        }
      }
    },
    interaction: { mode: 'nearest', intersect: true },
    onHover: (_, elements) => {
      if (!onPointHover) return;
      if (elements && elements.length > 0) {
        const idx = elements[0].index;
        onPointHover(data[idx]);
      }
    },
  };

  return <Line data={chartData} options={options} />;
};

export default PredictionChart;
