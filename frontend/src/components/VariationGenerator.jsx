import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  RefreshCw, Shuffle, Copy, Star, Zap,
  Music2, Sliders, Layers, ArrowRight,
  Save, Download, Eye, Grid, List
} from 'lucide-react';
import './VariationGenerator.css';

function VariationGenerator() {
  const [currentTrack, setCurrentTrack] = useState({
    id: 1,
    name: 'Epic Battle Theme',
    duration: '3:45',
    variations: 8
  });

  const [variations, setVariations] = useState([
    { id: 1, name: 'Original', icon: '🎵', mood: 'epic', tempo: 140, selected: true },
    { id: 2, name: 'Intense', icon: '🔥', mood: 'intense', tempo: 160, selected: false },
    { id: 3, name: 'Ambient', icon: '🌙', mood: 'ambient', tempo: 80, selected: false },
    { id: 4, name: 'orchestral', icon: '🎻', mood: 'dramatic', tempo: 120, selected: false },
    { id: 5, name: 'Electronic', icon: '⚡', mood: 'electronic', tempo: 150, selected: false },
    { id: 6, name: 'Minimal', icon: '○', mood: 'minimal', tempo: 100, selected: false },
  ]);

  const [generationMode, setGenerationMode] = useState('auto');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [viewMode, setViewMode] = useState('grid');
  const [selectedVariation, setSelectedVariation] = useState(1);
  const [parameters, setParameters] = useState({
    tempo: 50,
    intensity: 50,
    darkness: 50,
    warmth: 50,
    complexity: 50
  });

  const generateVariations = useCallback(async (count = 5) => {
    setIsGenerating(true);
    setGenerationProgress(0);

    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 300));
      setGenerationProgress(i);
    }

    const newVariations = Array.from({ length: count }, (_, i) => ({
      id: Date.now() + i,
      name: `Variation ${variations.length + i + 1}`,
      icon: ['🎼', '🎹', '🎺', '🥁', '🎸'][Math.floor(Math.random() * 5)],
      mood: ['epic', 'ambient', 'dark', 'hopeful', 'mysterious'][Math.floor(Math.random() * 5)],
      tempo: 80 + Math.floor(Math.random() * 80),
      selected: false
    }));

    setVariations(prev => [...prev, ...newVariations]);
    setIsGenerating(false);
  }, [variations.length]);

  const quickVariation = async () => {
    setIsGenerating(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setVariations(prev => prev.map(v => 
      v.id === selectedVariation 
        ? { ...v, name: `${v.name} (v2)`, tempo: v.tempo + (Math.random() > 0.5 ? 10 : -10) }
        : v
    ));
    
    setIsGenerating(false);
  };

  const randomizeAll = async () => {
    setIsGenerating(true);
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setVariations(prev => prev.map(v => ({
      ...v,
      tempo: 80 + Math.floor(Math.random() * 80),
      mood: ['epic', 'ambient', 'dark', 'hopeful', 'mysterious', 'intense'][Math.floor(Math.random() * 6)]
    })));
    
    setIsGenerating(false);
  };

  const handleParameterChange = (key, value) => {
    setParameters(prev => ({ ...prev, [key]: parseInt(value) }));
  };

  return (
    <div className="variation-generator">
      <div className="variation-header">
        <div className="variation-header-content">
          <Layers className="header-icon" />
          <div>
            <h2>Infinite Variations</h2>
            <p>Generate unlimited variations of your tracks with AI-powered transformations</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="view-toggle">
            <button 
              className={`view-btn ${viewMode === 'grid' ? 'active' : ''}`}
              onClick={() => setViewMode('grid')}
            >
              <Grid size={18} />
            </button>
            <button 
              className={`view-btn ${viewMode === 'list' ? 'active' : ''}`}
              onClick={() => setViewMode('list')}
            >
              <List size={18} />
            </button>
          </div>
        </div>
      </div>

      <div className="variation-content">
        <div className="variation-main">
          {/* Generation Controls */}
          <div className="generation-controls card-glass">
            <div className="controls-header">
              <Zap size={20} />
              <h3>Generate Variations</h3>
            </div>

            <div className="generation-modes">
              <button
                className={`mode-btn ${generationMode === 'auto' ? 'active' : ''}`}
                onClick={() => setGenerationMode('auto')}
              >
                <Star size={18} />
                Auto
              </button>
              <button
                className={`mode-btn ${generationMode === 'custom' ? 'active' : ''}`}
                onClick={() => setGenerationMode('custom')}
              >
                <Sliders size={18} />
                Custom
              </button>
              <button
                className={`mode-btn ${generationMode === 'smart' ? 'active' : ''}`}
                onClick={() => setGenerationMode('smart')}
              >
                <BrainIcon size={18} />
                Smart Mix
              </button>
            </div>

            {generationMode === 'custom' && (
              <div className="parameters-section">
                <h4>Variation Parameters</h4>
                
                <div className="parameters-grid">
                  <div className="parameter">
                    <label>Tempo</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={parameters.tempo}
                      onChange={(e) => handleParameterChange('tempo', e.target.value)}
                      className="param-slider"
                    />
                    <span className="param-value">{parameters.tempo}%</span>
                  </div>

                  <div className="parameter">
                    <label>Intensity</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={parameters.intensity}
                      onChange={(e) => handleParameterChange('intensity', e.target.value)}
                      className="param-slider"
                    />
                    <span className="param-value">{parameters.intensity}%</span>
                  </div>

                  <div className="parameter">
                    <label>Darkness</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={parameters.darkness}
                      onChange={(e) => handleParameterChange('darkness', e.target.value)}
                      className="param-slider"
                    />
                    <span className="param-value">{parameters.darkness}%</span>
                  </div>

                  <div className="parameter">
                    <label>Warmth</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={parameters.warmth}
                      onChange={(e) => handleParameterChange('warmth', e.target.value)}
                      className="param-slider"
                    />
                    <span className="param-value">{parameters.warmth}%</span>
                  </div>

                  <div className="parameter">
                    <label>Complexity</label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={parameters.complexity}
                      onChange={(e) => handleParameterChange('complexity', e.target.value)}
                      className="param-slider"
                    />
                    <span className="param-value">{parameters.complexity}%</span>
                  </div>
                </div>
              </div>
            )}

            <div className="action-buttons">
              <button
                className="generate-btn primary"
                onClick={() => generateVariations(5)}
                disabled={isGenerating}
              >
                {isGenerating ? (
                  <>
                    <div className="spinner-small"></div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    Generate 5 Variations
                  </>
                )}
              </button>

              <button
                className="generate-btn secondary"
                onClick={quickVariation}
                disabled={isGenerating}
              >
                <RefreshCw size={18} />
                Quick Variation
              </button>

              <button
                className="generate-btn tertiary"
                onClick={randomizeAll}
                disabled={isGenerating}
              >
                <Shuffle size={18} />
                Randomize All
              </button>
            </div>

            {/* Progress Bar */}
            <AnimatePresence>
              {isGenerating && (
                <motion.div
                  className="generation-progress"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${generationProgress}%` }}
                    />
                  </div>
                  <p className="progress-text">
                    {generationProgress < 30 && 'Analyzing original track...'}
                    {generationProgress >= 30 && generationProgress < 60 && 'Applying transformations...'}
                    {generationProgress >= 60 && generationProgress < 90 && 'Rendering variations...'}
                    {generationProgress >= 90 && 'Finalizing...'}
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Variations Display */}
          <div className="variations-display card-glass">
            <div className="display-header">
              <Music2 size={20} />
              <h3>Current Variations ({variations.length})</h3>
              <span className="track-name">{currentTrack.name}</span>
            </div>

            <div className={`variations-${viewMode}`}>
              <AnimatePresence>
                {variations.map((variation, index) => (
                  <motion.div
                    key={variation.id}
                    className={`variation-card ${selectedVariation === variation.id ? 'selected' : ''}`}
                    onClick={() => setSelectedVariation(variation.id)}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="variation-icon">{variation.icon}</div>
                    
                    <div className="variation-info">
                      <h4>{variation.name}</h4>
                      <div className="variation-meta">
                        <span className="mood-tag">{variation.mood}</span>
                        <span className="tempo-tag">{variation.tempo} BPM</span>
                      </div>
                    </div>

                    <div className="variation-actions">
                      <button className="action-icon" title="Preview">
                        <Eye size={14} />
                      </button>
                      <button className="action-icon" title="Save">
                        <Save size={14} />
                      </button>
                      <button className="action-icon" title="Export">
                        <Download size={14} />
                      </button>
                    </div>

                    {selectedVariation === variation.id && (
                      <div className="selected-indicator">
                        <ArrowRight size={16} />
                      </div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* Compare View */}
          <div className="compare-section card-glass">
            <div className="compare-header">
              <Copy size={20} />
              <h3>Compare Variations</h3>
            </div>

            <div className="compare-controls">
              <select className="compare-select">
                {variations.map(v => (
                  <option key={v.id} value={v.id}>{v.name}</option>
                ))}
              </select>
              
              <span className="compare-arrow">
                <ArrowRight size={20} />
              </span>
              
              <select className="compare-select">
                {variations.map(v => (
                  <option key={v.id} value={v.id}>{v.name}</option>
                ))}
              </select>

              <button className="btn btn-primary">
                <Zap size={16} />
                Generate Blend
              </button>
            </div>

            <div className="compare-visualization">
              <div className="waveform-comparison">
                {variations.slice(0, 4).map((v, i) => (
                  <div key={v.id} className="waveform-row">
                    <span className="waveform-label">{v.name}</span>
                    <div className="waveform-bars">
                      {[...Array(30)].map((_, j) => (
                        <div
                          key={j}
                          className="waveform-bar"
                          style={{
                            height: `${Math.random() * 100}%`,
                            opacity: 0.3 + (i * 0.2)
                          }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <aside className="variation-sidebar">
          <div className="sidebar-section card-glass">
            <h4>Quick Actions</h4>
            
            <div className="quick-actions">
              <button className="quick-action-btn">
                <Star size={18} />
                <span>Mark as Favorite</span>
              </button>
              <button className="quick-action-btn">
                <Copy size={18} />
                <span>Duplicate</span>
              </button>
              <button className="quick-action-btn">
                <Music2 size={18} />
                <span>Apply to Project</span>
              </button>
              <button className="quick-action-btn">
                <Download size={18} />
                <span>Export All</span>
              </button>
            </div>
          </div>

          <div className="sidebar-section card-glass">
            <h4>Popular Presets</h4>
            
            <div className="preset-list">
              {['Epic Remaster', 'Cinematic Dark', 'Uplifting Mix', 'Minimalist', 'Orchestral'].map((preset, i) => (
                <button key={i} className="preset-btn">
                  <Star size={14} />
                  {preset}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-section card-glass">
            <h4>Statistics</h4>
            
            <div className="stats-list">
              <div className="stat-row">
                <span>Total Variations</span>
                <span className="stat-value">{variations.length}</span>
              </div>
              <div className="stat-row">
                <span>Favorites</span>
                <span className="stat-value">3</span>
              </div>
              <div className="stat-row">
                <span>Exported</span>
                <span className="stat-value">2</span>
              </div>
              <div className="stat-row">
                <span>Avg. Tempo</span>
                <span className="stat-value">
                  {Math.round(variations.reduce((acc, v) => acc + v.tempo, 0) / variations.length)} BPM
                </span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

// Brain icon component
function BrainIcon({ size }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
    </svg>
  );
}

export default VariationGenerator;

