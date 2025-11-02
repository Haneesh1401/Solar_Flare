const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');
const mongoose = require('mongoose');

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

/* MongoDB connection (optional - commented out for now)
// mongoose.connect('mongodb://localhost:27017/solar_flare_db', {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => console.log('MongoDB connected'))
  .catch(err => console.log(err));
*/

// Routes
app.get('/api/predict', async (req, res) => {
  // Mock prediction data for testing (alternative method since Python integration is failing)
  const mockPredictions = {
    c: 0.25,
    m: 0.15,
    x: 0.05,
    confidence: 0.85,
    accuracy: 0.92,
    timestamp: new Date().toISOString(),
    predicted_class: 'c'
  };
  res.json(mockPredictions);
});

app.get('/api/data', (req, res) => {
  // Mock timeline data for testing
  const mockTimeline = [
    { time: 'Now', c: 0.25, m: 0.15, x: 0.05 },
    { time: '24h', c: 0.30, m: 0.20, x: 0.10 },
    { time: '48h', c: 0.35, m: 0.25, x: 0.15 }
  ];
  res.json({ timeline: mockTimeline });
});

app.get('/api/models', (req, res) => {
  // Model info
  res.json({ accuracy: 0.92, confidence: 0.85 });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
