import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Heart, Zap, TrendingUp, TrendingDown,
  Activity, Thermometer, Wind, Moon,
  Sun, Cloud, Droplets, Flame,
  RefreshCw, Save, Sliders
} from 'lucide-react';
import './EmotionEngine.css';

function EmotionEngine() {
  const [emotions, setEmotions] = useState({
    tension: 50,
    release: 50,
    intensity: 50,
    warmth: 50,
    darkness: 50,
    energy: 50,
    hope: 50,
    dread: 50
  });

  const [selectedPreset, setSelectedPreset] = useState('custom');
  const [isAnimating, setIsAnimating] = useState(false);
  const [waveformData, setWaveformData] = useState([]);
  const [targetEmotion, setTargetEmotion] = useState(null);
  const [transitionProgress, setTransitionProgress] = useState(0);

  const presets = {
    'epic-battle': {
      tension: 90,
      release: 30,
      intensity: 95,
      warmth: 40,
      darkness: 60,
      energy: 95,
      hope: 30,
      dread: 70
    },
    'romantic': {
      tension: 20,
      release: 80,
      intensity: 30,
      warmth: 90,
      darkness: 10,
      energy: 30,
      hope: 80,
      dread: 5
    },
    'mysterious': {
      tension: 70,
      release: 40,
      intensity: 40,
      warmth: 30,
      darkness: 80,
      energy: 30,
      hope: 20,
      dread: 60
    },
    'triumphant': {
      tension: 30,
      release: 70,
      intensity: 80,
      warmth: 70,
      darkness: 20,
      energy: 85,
      hope: 95,
      dread: 5
    },
    'horror': {
      tension: 95,
      release: 10,
      intensity: 90,
      warmth: 10,
      darkness: 100,
      energy: 60,
      hope: 5,
      dread: 95
    },
    'peaceful': {
      tension: 10,
      release: 90,
      intensity: 20,
      warmth: 80,
      darkness: 10,
      energy: 20,
      hope: 70,
      dread: 5
    }
  };

  const handleEmotionChange = (key, value) => {
    setEmotions(prev => ({
      ...prev,
      [key]: parseInt(value)
    }));
    setSelectedPreset('custom');
  };

  const applyPreset = useCallback((presetName) => {
    const preset = presets[presetName];
    if (preset) {
      setIsAnimating(true);
      setSelectedPreset(presetName);
      
      // Animate transition
      let progress = 0;
      const interval = setInterval(() => {
        progress += 2;
        setTransitionProgress(progress);
        
        setEmotions(prev => {
          const newEmotions = {};
          Object.keys(preset).forEach(key => {
            newEmotions[key] = Math.round(
              prev[key] + (preset[key] - prev[key]) * (progress / 100)
            );
          });
          return newEmotions;
        });
        
        if (progress >= 100) {
          clearInterval(interval);
          setIsAnimating(false);
          setTransitionProgress(0);
        }
      }, 20);
    }
  }, [presets]);

  const randomize = () => {
    const randomPreset = Object.keys(presets)[
      Math.floor(Math.random() * Object.keys(presets).length)
    ];
    applyPreset(randomPreset);
  };

  // Generate waveform visualization data
  useEffect(() => {
    const generateWaveform = () => {
      const data = [];
      const baseAmplitude = emotions.energy / 100;
      
      for (let i = 0; i < 100; i++) {
        const tensionFactor = emotions.tension / 100;
        const intensityFactor = emotions.intensity / 100;
        const noise = Math.random() * 0.3 + 0.7;
        const wave = Math.sin(i * 0.1 + Date.now() * 0.001) * 0.5 + 0.5;
        
        data.push({
          height: baseAmplitude * intensityFactor * noise * wave * 100,
          tension: tensionFactor
        });
      }
      
      setWaveformData(data);
    };
    
    const interval = setInterval(generateWaveform, 100);
    return () => clearInterval(interval);
  }, [emotions]);

  const getEmotionColor = (value) => {
    if (value < 33) return 'var(--color-cyan-primary)';
    if (value < 66) return 'var(--color-purple-primary)';
    return 'var(--color-gold-primary)';
  };

  const getEmotionIcon = (key) => {
    const icons = {
      tension: <Activity size={18} />,
      release: <Wind size={18} />,
      intensity: <Flame size={18} />,
      warmth: <Sun size={18} />,
      darkness: <Moon size={18} />,
      energy: <Zap size={18} />,
      hope: <TrendingUp size={18} />,
      dread: <TrendingDown size={18} />
    };
    return icons[key];
  };

  const getEmotionLabel = (key) => {
    const labels = {
      tension: 'Tension',
      release: 'Release',
      intensity: 'Intensity',
      warmth: 'Warmth',
      darkness: 'Darkness',
      energy: 'Energy',
      hope: 'Hope',
      dread: 'Dread'
    };
    return labels[key];
  };

  const getEmotionDescription = (key, value) => {
    const descriptions = {
      tension: value < 33 ? 'Calm & relaxed' : value < 66 ? 'Building suspense' : 'Maximum tension',
      release: value < 33 ? 'Constrained' : value < 66 ? 'Balanced flow' : 'Free expression',
      intensity: value < 33 ? 'Subtle' : value < 66 ? 'Moderate' : 'Explosive',
      warmth: value < 33 ? 'Cold' : value < 66 ? 'Neutral' : 'Warm & inviting',
      darkness: value < 33 ? 'Light' : value < 66 ? 'Balanced' : 'Dark & ominous',
      energy: value < 33 ? 'Low energy' : value < 66 ? 'Moderate' : 'High energy',
      hope: value < 33 ? 'Despairing' : value < 66 ? 'Uncertain' : 'Hopeful',
      dread: value < 33 ? 'Safe' : value < 66 ? 'Uneasy' : 'Terrifying'
    };
    return descriptions[key];
  };

  return (
    <div className="emotion-engine">
      <div className="emotion-header">
        <div className="emotion-header-content">
          <Heart className="header-icon" />
          <div>
            <h2>Emotion Engine</h2>
            <p>Fine-tune emotional characteristics for adaptive music generation</p>
          </div>
        </div>

        <div className="header-actions">
          <button className="action-btn" onClick={randomize}>
            <RefreshCw size={18} />
            Randomize
          </button>
          <button className="action-btn">
            <Save size={18} />
            Save Preset
          </button>
        </div>
      </div>

      <div className="emotion-content">
        {/* Presets Bar */}
        <div className="presets-bar card-glass">
          <div className="presets-label">
            <Sliders size={16} />
            Quick Presets
          </div>
          <div className="presets-list">
            {Object.keys(presets).map(preset => (
              <button
                key={preset}
                className={`preset-btn ${selectedPreset === preset ? 'active' : ''}`}
                onClick={() => applyPreset(preset)}
              >
                {preset.split('-').map(word => 
                  word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' ')}
              </button>
            ))}
          </div>
        </div>

        <div className="emotion-main">
          {/* Emotion Sliders */}
          <div className="emotion-sliders card-glass">
            <h3 className="section-title">Emotion Parameters</h3>
            
            <div className="sliders-grid">
              {Object.keys(emotions).map(key => (
                <motion.div 
                  key={key}
                  className="emotion-slider"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: Object.keys(emotions).indexOf(key) * 0.05 }}
                >
                  <div className="slider-header">
                    <div className="slider-icon" style={{ color: getEmotionColor(emotions[key]) }}>
                      {getEmotionIcon(key)}
                    </div>
                    <div className="slider-info">
                      <span className="slider-label">{getEmotionLabel(key)}</span>
                      <span className="slider-description">
                        {getEmotionDescription(key, emotions[key])}
                      </span>
                    </div>
                    <span className="slider-value" style={{ color: getEmotionColor(emotions[key]) }}>
                      {emotions[key]}
                    </span>
                  </div>
                  
                  <div className="slider-track">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={emotions[key]}
                      onChange={(e) => handleEmotionChange(key, e.target.value)}
                      className="emotion-range"
                      style={{
                        background: `linear-gradient(to right, 
                          var(--color-cyan-primary) 0%, 
                          var(--color-cyan-primary) 33%, 
                          var(--color-purple-primary) 33%, 
                          var(--color-purple-primary) 66%, 
                          var(--color-gold-primary) 66%, 
                          var(--color-gold-primary) 100%)`
                      }}
                    />
                    <div 
                      className="slider-fill"
                      style={{ 
                        width: `${emotions[key]}%`,
                        background: getEmotionColor(emotions[key])
                      }}
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </div>

          {/* Emotion Visualization */}
          <div className="emotion-visualization card-glass">
            <h3 className="section-title">Real-time Visualization</h3>
            
            <div className="visualization-container">
              {/* 3D-style Waveform */}
              <div className="waveform-3d">
                <div className="waveform-bars">
                  {waveformData.map((bar, i) => (
                    <motion.div
                      key={i}
                      className="waveform-bar-3d"
                      style={{
                        height: `${bar.height}%`,
                        background: bar.tension > 0.5 
                          ? `linear-gradient(to top, var(--color-purple-primary), var(--color-gold-primary))`
                          : `linear-gradient(to top, var(--color-cyan-primary), var(--color-purple-primary))`,
                        transform: `perspective(500px) rotateX(${bar.tension * 30}deg)`
                      }}
                      animate={{
                        height: [`${bar.height}%`, `${bar.height * 1.2}%`, `${bar.height}%`]
                      }}
                      transition={{
                        duration: 0.5,
                        delay: i * 0.01
                      }}
                    />
                  ))}
                </div>
              </div>

              {/* Emotion Radar */}
              <div className="emotion-radar">
                <svg viewBox="0 0 200 200" className="radar-chart">
                  {/* Radar grid */}
                  {[0.25, 0.5, 0.75, 1].map(scale => (
                    <polygon
                      key={scale}
                      points={calculatePolygonPoints(8, 80 * scale, 100, 100)}
                      fill="none"
                      stroke="rgba(255,255,255,0.1)"
                      strokeWidth="1"
                    />
                  ))}
                  
                  {/* Axis lines */}
                  {Object.keys(emotions).map((_, i) => {
                    const angle = (i * 2 * Math.PI / 8) - Math.PI / 2;
                    const x2 = 100 + 80 * Math.cos(angle);
                    const y2 = 100 + 80 * Math.sin(angle);
                    return (
                      <line
                        key={i}
                        x1="100"
                        y1="100"
                        x2={x2}
                        y2={y2}
                        stroke="rgba(255,255,255,0.1)"
                        strokeWidth="1"
                      />
                    );
                  })}
                  
                  {/* Emotion polygon */}
                  <polygon
                    points={calculatePolygonPoints(8, 80, 100, 100, Object.values(emotions).map(v => v / 100))}
                    fill="rgba(212, 175, 55, 0.3)"
                    stroke="var(--color-gold-primary)"
                    strokeWidth="2"
                  />
                  
                  {/* Data points */}
                  {Object.values(emotions).map((value, i) => {
                    const angle = (i * 2 * Math.PI / 8) - Math.PI / 2;
                    const radius = (value / 100) * 80;
                    const x = 100 + radius * Math.cos(angle);
                    const y = 100 + radius * Math.sin(angle);
                    return (
                      <circle
                        key={i}
                        cx={x}
                        cy={y}
                        r="4"
                        fill={getEmotionColor(value)}
                        className="radar-point"
                      />
                    );
                  })}
                </svg>
                
                {/* Labels */}
                <div className="radar-labels">
                  {Object.keys(emotions).map((key, i) => {
                    const angle = (i * 2 * Math.PI / 8) - Math.PI / 2;
                    const x = 100 + 100 * Math.cos(angle);
                    const y = 100 + 100 * Math.sin(angle);
                    return (
                      <span
                        key={key}
                        className="radar-label"
                        style={{
                          left: `${x}%`,
                          top: `${y}%`,
                          color: getEmotionColor(emotions[key])
                        }}
                      >
                        {getEmotionLabel(key)}
                      </span>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Emotion Summary */}
            <div className="emotion-summary">
              <div className="summary-item">
                <span className="summary-label">Dominant Mood</span>
                <span className="summary-value" style={{ color: getEmotionColor(emotions.intensity) }}>
                  {emotions.intensity > 70 ? 'Intense' : emotions.intensity > 40 ? 'Moderate' : 'Subtle'}
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Emotional Range</span>
                <span className="summary-value" style={{ color: getEmotionColor(emotions.tension) }}>
                  {Math.abs(emotions.tension - 50) > 30 ? 'Extreme' : 'Balanced'}
                </span>
              </div>
              <div className="summary-item">
                <span className="summary-label">Overall Tone</span>
                <span className="summary-value" style={{ color: getEmotionColor(emotions.warmth) }}>
                  {emotions.warmth > 60 ? 'Warm' : emotions.warmth < 40 ? 'Cold' : 'Neutral'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar - Quick Adjust */}
        <aside className="emotion-sidebar">
          <div className="sidebar-section card-glass">
            <h4>Quick Adjust</h4>
            
            <div className="quick-adjust-group">
              <label>Dramatic Arc</label>
              <div className="arc-buttons">
                <button 
                  className="arc-btn"
                  onClick={() => {
                    setEmotions(prev => ({ ...prev, tension: 90, intensity: 80, dread: 60 }));
                  }}
                >
                  Rising
                </button>
                <button 
                  className="arc-btn"
                  onClick={() => {
                    setEmotions(prev => ({ ...prev, tension: 40, intensity: 40, release: 80 }));
                  }}
                >
                  Falling
                </button>
                <button 
                  className="arc-btn"
                  onClick={() => {
                    setEmotions(prev => ({ ...prev, tension: 70, intensity: 70, hope: 50 }));
                  }}
                >
                  Climax
                </button>
              </div>
            </div>

            <div className="quick-adjust-group">
              <label>Intensity Boost</label>
              <button 
                className="boost-btn"
                onClick={() => {
                  setEmotions(prev => ({
                    ...prev,
                    intensity: Math.min(100, prev.intensity + 20),
                    energy: Math.min(100, prev.energy + 20),
                    tension: Math.min(100, prev.tension + 10)
                  }));
                }}
              >
                <Zap size={16} />
                +20% Intensity
              </button>
            </div>

            <div className="quick-adjust-group">
              <label>Calm Down</label>
              <button 
                className="calm-btn"
                onClick={() => {
                  setEmotions(prev => ({
                    ...prev,
                    intensity: Math.max(0, prev.intensity - 20),
                    energy: Math.max(0, prev.energy - 20),
                    tension: Math.max(0, prev.tension - 20),
                    dread: Math.max(0, prev.dread - 10)
                  }));
                }}
              >
                <Wind size={16} />
                -20% Intensity
              </button>
            </div>
          </div>

          <div className="sidebar-section card-glass">
            <h4>Generation Preview</h4>
            <p className="preview-description">
              Based on your emotion settings, the AI will generate music with the following characteristics:
            </p>
            
            <ul className="preview-list">
              <li>
                <strong>Tempo:</strong> {Math.round(60 + (emotions.energy / 100) * 100)} BPM
              </li>
              <li>
                <strong>Key:</strong> {emotions.warmth > 50 ? 'Major' : 'Minor'}
              </li>
              <li>
                <strong>Dynamics:</strong> {emotions.intensity > 70 ? 'Wide range' : 'Controlled'}
              </li>
              <li>
                <strong>Texture:</strong> {emotions.tension > 60 ? 'Dense' : 'Sparse'}
              </li>
              <li>
                <strong>Harmony:</strong> {emotions.darkness > 50 ? 'Complex' : 'Simple'}
              </li>
            </ul>

            <button className="btn btn-primary btn-block">
              <Heart size={18} />
              Generate with These Emotions
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}

// Helper function for radar chart polygon
function calculatePolygonPoints(sides, radius, cx, cy, scales = []) {
  const points = [];
  for (let i = 0; i < sides; i++) {
    const angle = (i * 2 * Math.PI / sides) - Math.PI / 2;
    const scale = scales[i] || 1;
    const r = radius * scale;
    const x = cx + r * Math.cos(angle);
    const y = cy + r * Math.sin(angle);
    points.push(`${x},${y}`);
  }
  return points.join(' ');
}

export default EmotionEngine;

