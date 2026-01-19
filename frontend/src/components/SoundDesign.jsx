import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Wind, Waves, Sparkles, Zap, Music2,
  Volume2, Sliders, Layers, Snowflake, Flame,
  Radio, Disc, Headphones, Leaf
} from 'lucide-react';
import './SoundDesign.css';

function SoundDesign() {
  const [layers, setLayers] = useState([
    { 
      id: 1, 
      name: 'Wind', 
      icon: <Wind size={20} />, 
      volume: 60, 
      muted: false,
      category: 'ambient',
      preset: 'Gentle Breeze'
    },
    { 
      id: 2, 
      name: 'Rain', 
      icon: <Waves size={20} />, 
      volume: 45, 
      muted: false,
      category: 'ambient',
      preset: 'Light Rain'
    },
    { 
      id: 3, 
      name: 'Thunder', 
      icon: <Flame size={20} />, 
      volume: 70, 
      muted: true,
      category: 'weather',
      preset: 'Distant Roll'
    },
    { 
      id: 4, 
      name: 'Forest', 
      icon: <Leaf size={20} />, 
      volume: 50, 
      muted: false,
      category: 'nature',
      preset: 'Morning Ambience'
    },
    { 
      id: 5, 
      name: 'Cinematic Impact', 
      icon: <Zap size={20} />, 
      volume: 80, 
      muted: false,
      category: 'fx',
      preset: 'Deep Boom'
    },
    { 
      id: 6, 
      name: 'Space Drone', 
      icon: <Radio size={20} />, 
      volume: 40, 
      muted: false,
      category: 'texture',
      preset: 'Deep Space'
    },
  ]);

  const [masterVolume, setMasterVolume] = useState(75);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [isPlaying, setIsPlaying] = useState(true);

  const presets = {
    ambient: ['Gentle Breeze', 'Light Rain', 'Ocean Waves', 'Cave Drip'],
    nature: ['Morning Ambience', 'Night Forest', 'Jungle', 'River Flow'],
    weather: ['Distant Roll', 'Storm Close', 'Wind Howl', 'Lightning'],
    fx: ['Deep Boom', 'Riser', 'Impact', 'Sub Drop'],
    texture: ['Deep Space', 'Ethereal', 'Mechanical', 'Digital']
  };

  const handleVolumeChange = (id, value) => {
    setLayers(prev => prev.map(layer => 
      layer.id === id ? { ...layer, volume: parseInt(value) } : layer
    ));
  };

  const toggleMute = (id) => {
    setLayers(prev => prev.map(layer => 
      layer.id === id ? { ...layer, muted: !layer.muted } : layer
    ));
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'ambient': return <Wind size={14} />;
      case 'nature': return <Leaf size={14} />;
      case 'weather': return <Waves size={14} />;
      case 'fx': return <Zap size={14} />;
      case 'texture': return <Disc size={14} />;
      default: return <Sparkles size={14} />;
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'ambient': return '#06b6d4';
      case 'nature': return '#10b981';
      case 'weather': return '#6366f1';
      case 'fx': return '#f59e0b';
      case 'texture': return '#8b5cf6';
      default: return '#6b7280';
    }
  };

  return (
    <div className="sound-design">
      <div className="sound-header">
        <div className="sound-header-content">
          <Sparkles className="header-icon" />
          <div>
            <h2>Cinematic Sound Design</h2>
            <p>Layer textures, foley, and atmospheric effects</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="master-volume">
            <Volume2 size={18} />
            <input
              type="range"
              min="0"
              max="100"
              value={masterVolume}
              onChange={(e) => setMasterVolume(e.target.value)}
              className="master-slider"
            />
            <span>{masterVolume}%</span>
          </div>

          <button 
            className={`play-btn ${isPlaying ? 'playing' : ''}`}
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
        </div>
      </div>

      <div className="sound-content">
        {/* Main Layer Mixer */}
        <div className="layer-mixer card-glass">
          <div className="mixer-header">
            <Layers size={20} />
            <h3>Sound Layers</h3>
            <span className="layer-count">{layers.length} layers</span>
          </div>

          <div className="layers-grid">
            {layers.map((layer) => (
              <motion.div
                key={layer.id}
                className={`layer-card ${selectedLayer === layer.id ? 'selected' : ''} ${layer.muted ? 'muted' : ''}`}
                onClick={() => setSelectedLayer(layer.id)}
                style={{ '--layer-color': getCategoryColor(layer.category) }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="layer-header">
                  <div 
                    className="layer-icon"
                    style={{ backgroundColor: getCategoryColor(layer.category) }}
                  >
                    {layer.icon}
                  </div>
                  <div className="layer-info">
                    <h4>{layer.name}</h4>
                    <span className="layer-preset">{layer.preset}</span>
                  </div>
                  <button 
                    className={`mute-btn ${layer.muted ? 'active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleMute(layer.id);
                    }}
                  >
                    {layer.muted ? <Volume2 size={14} /> : <Volume2 size={14} />}
                  </button>
                </div>

                <div className="layer-visual">
                  <div className="waveform-bars">
                    {[...Array(20)].map((_, i) => (
                      <div
                        key={i}
                        className="waveform-bar"
                        style={{
                          height: layer.muted ? '10%' : `${20 + Math.random() * 80}%`,
                          animationDelay: `${i * 0.05}s`,
                          opacity: layer.muted ? 0.3 : 1
                        }}
                      />
                    ))}
                  </div>
                </div>

                <div className="layer-controls">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={layer.volume}
                    onChange={(e) => handleVolumeChange(layer.id, e.target.value)}
                    className="layer-slider"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <span className="layer-volume">{layer.volume}%</span>
                </div>

                <div className="layer-category">
                  {getCategoryIcon(layer.category)}
                  <span>{layer.category}</span>
                </div>
              </motion.div>
            ))}

            {/* Add Layer Button */}
            <button className="add-layer-btn">
              <Sparkles size={32} />
              <span>Add Layer</span>
            </button>
          </div>
        </div>

        {/* Layer Details Panel */}
        <div className="layer-details card-glass">
          <div className="details-header">
            <Sliders size={20} />
            <h3>Layer Settings</h3>
          </div>

          {selectedLayer ? (
            (() => {
              const layer = layers.find(l => l.id === selectedLayer);
              if (!layer) return null;
              
              return (
                <>
                  <div className="selected-layer-info">
                    <div 
                      className="layer-avatar"
                      style={{ backgroundColor: getCategoryColor(layer.category) }}
                    >
                      {layer.icon}
                    </div>
                    <div>
                      <h4>{layer.name}</h4>
                      <span className="category-badge" style={{ backgroundColor: getCategoryColor(layer.category) }}>
                        {layer.category}
                      </span>
                    </div>
                  </div>

                  {/* Presets */}
                  <div className="setting-section">
                    <label>Presets</label>
                    <div className="preset-grid">
                      {presets[layer.category].map((preset, i) => (
                        <button 
                          key={i}
                          className={`preset-btn ${layer.preset === preset ? 'active' : ''}`}
                          onClick={() => setLayers(prev => prev.map(l => 
                            l.id === layer.id ? { ...l, preset } : l
                          ))}
                        >
                          {preset}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* EQ Controls */}
                  <div className="setting-section">
                    <label>EQ</label>
                    <div className="eq-controls">
                      <div className="eq-knob">
                        <span>LOW</span>
                        <div className="knob"></div>
                        <input type="range" min="-12" max="12" defaultValue="0" className="eq-slider" />
                      </div>
                      <div className="eq-knob">
                        <span>MID</span>
                        <div className="knob"></div>
                        <input type="range" min="-12" max="12" defaultValue="0" className="eq-slider" />
                      </div>
                      <div className="eq-knob">
                        <span>HIGH</span>
                        <div className="knob"></div>
                        <input type="range" min="-12" max="12" defaultValue="0" className="eq-slider" />
                      </div>
                    </div>
                  </div>

                  {/* Effects */}
                  <div className="setting-section">
                    <label>Effects</label>
                    <div className="effects-list">
                      <div className="effect-row">
                        <span>Reverb</span>
                        <input type="range" min="0" max="100" defaultValue="30" className="effect-slider" />
                      </div>
                      <div className="effect-row">
                        <span>Delay</span>
                        <input type="range" min="0" max="100" defaultValue="10" className="effect-slider" />
                      </div>
                      <div className="effect-row">
                        <span>Filter</span>
                        <input type="range" min="0" max="100" defaultValue="50" className="effect-slider" />
                      </div>
                    </div>
                  </div>
                </>
              );
            })()
          ) : (
            <div className="no-selection">
              <Headphones size={48} />
              <p>Select a layer to edit settings</p>
            </div>
          )}
        </div>

        {/* Quick Presets */}
        <div className="quick-presets card-glass">
          <h3>Quick Presets</h3>
          
          <div className="preset-cards">
            <div className="preset-card">
              <div className="preset-icon" style={{ background: 'linear-gradient(135deg, #1a1a26, #2d2d44)' }}>
                <Wind size={24} />
              </div>
              <div className="preset-info">
                <h4>Atmospheric Wind</h4>
                <p>Light breeze with subtle movement</p>
              </div>
              <button className="apply-preset-btn">Apply</button>
            </div>

            <div className="preset-card">
              <div className="preset-icon" style={{ background: 'linear-gradient(135deg, #0f172a, #1e3a5f)' }}>
                <Waves size={24} />
              </div>
              <div className="preset-info">
                <h4>Ocean Storm</h4>
                <p>Powerful waves and rain</p>
              </div>
              <button className="apply-preset-btn">Apply</button>
            </div>

            <div className="preset-card">
              <div className="preset-icon" style={{ background: 'linear-gradient(135deg, #1e1b4b, #312e81)' }}>
                <Disc size={24} />
              </div>
              <div className="preset-info">
                <h4>Sci-Fi Ambience</h4>
                <p>Deep space textures</p>
              </div>
              <button className="apply-preset-btn">Apply</button>
            </div>

            <div className="preset-card">
              <div className="preset-icon" style={{ background: 'linear-gradient(135deg, #451a03, #78350f)' }}>
                <Flame size={24} />
              </div>
              <div className="preset-info">
                <h4>Epic Impact</h4>
                <p>Cinematic booms and rises</p>
              </div>
              <button className="apply-preset-btn">Apply</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SoundDesign;

