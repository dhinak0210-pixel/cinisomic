import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Volume2, VolumeX, Sliders, Music2, Headphones,
  Mic, Guitar, Piano, Drum, Violin, Trumpet,
  Waves, Activity, Filter, Crosshair
} from 'lucide-react';
import './StemControl.css';

function StemControl() {
  const [stems, setStems] = useState([
    { 
      id: 1, 
      name: 'Strings', 
      icon: <Violin size={20} />, 
      volume: 80, 
      pan: 0, 
      muted: false, 
      solo: false,
      color: '#d4af37',
      effects: { reverb: 30, delay: 10, eq: 50 }
    },
    { 
      id: 2, 
      name: 'Brass', 
      icon: <Trumpet size={20} />, 
      volume: 70, 
      pan: 10, 
      muted: false, 
      solo: false,
      color: '#ef4444',
      effects: { reverb: 40, delay: 5, eq: 60 }
    },
    { 
      id: 3, 
      name: 'Percussion', 
      icon: <Drum size={20} />, 
      volume: 85, 
      pan: 0, 
      muted: false, 
      solo: false,
      color: '#06b6d4',
      effects: { reverb: 20, delay: 0, eq: 70 }
    },
    { 
      id: 4, 
      name: 'Woodwinds', 
      icon: <Piano size={20} />, 
      volume: 60, 
      pan: -10, 
      muted: false, 
      solo: false,
      color: '#10b981',
      effects: { reverb: 35, delay: 15, eq: 45 }
    },
    { 
      id: 5, 
      name: 'Vocals', 
      icon: <Mic size={20} />, 
      volume: 90, 
      pan: 0, 
      muted: false, 
      solo: false,
      color: '#8b5cf6',
      effects: { reverb: 50, delay: 20, eq: 80 }
    },
    { 
      id: 6, 
      name: 'Synth Pad', 
      icon: <Music2 size={20} />, 
      volume: 55, 
      pan: 20, 
      muted: false, 
      solo: false,
      color: '#ec4899',
      effects: { reverb: 60, delay: 30, eq: 55 }
    },
  ]);

  const [selectedStem, setSelectedStem] = useState(null);
  const [masterVolume, setMasterVolume] = useState(80);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showEffects, setShowEffects] = useState(false);

  const handleVolumeChange = (id, value) => {
    setStems(prev => prev.map(stem => 
      stem.id === id ? { ...stem, volume: parseInt(value) } : stem
    ));
  };

  const handlePanChange = (id, value) => {
    setStems(prev => prev.map(stem => 
      stem.id === id ? { ...stem, pan: parseInt(value) } : stem
    ));
  };

  const toggleMute = (id) => {
    setStems(prev => prev.map(stem => 
      stem.id === id ? { ...stem, muted: !stem.muted } : stem
    ));
  };

  const toggleSolo = (id) => {
    setStems(prev => prev.map(stem => 
      stem.id === id ? { ...stem, solo: !stem.solo } : stem
    ));
  };

  const handleEffectChange = (id, effect, value) => {
    setStems(prev => prev.map(stem => 
      stem.id === id ? { 
        ...stem, 
        effects: { ...stem.effects, [effect]: parseInt(value) } 
      } : stem
    ));
  };

  const soloOne = stems.filter(s => s.solo).length === 1;

  return (
    <div className="stem-control">
      <div className="stem-header">
        <div className="stem-header-content">
          <Sliders className="header-icon" />
          <div>
            <h2>Stem Control Panel</h2>
            <p>Fine-tune individual instrument tracks with precision controls</p>
          </div>
        </div>

        <div className="master-controls">
          <div className="master-volume">
            <span className="master-label">Master</span>
            <div className="volume-slider">
              <input
                type="range"
                min="0"
                max="100"
                value={masterVolume}
                onChange={(e) => setMasterVolume(e.target.value)}
                className="master-slider"
              />
              <div 
                className="volume-fill"
                style={{ width: `${masterVolume}%` }}
              />
            </div>
            <span className="volume-value">{masterVolume}%</span>
          </div>

          <button 
            className={`play-btn ${isPlaying ? 'playing' : ''}`}
            onClick={() => setIsPlaying(!isPlaying)}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
        </div>
      </div>

      <div className="stem-content">
        {/* Main Stem Mixer */}
        <div className="stem-mixer card-glass">
          <div className="mixer-header">
            <Headphones size={20} />
            <h3>Stem Mixer</h3>
          </div>

          <div className="stems-grid">
            {stems.map((stem) => (
              <motion.div
                key={stem.id}
                className={`stem-channel ${selectedStem === stem.id ? 'selected' : ''} ${stem.muted ? 'muted' : ''} ${stem.solo ? 'solo' : ''}`}
                onClick={() => setSelectedStem(stem.id)}
                whileHover={{ scale: 1.02 }}
                style={{ '--stem-color': stem.color }}
              >
                <div className="channel-header">
                  <div className="stem-icon" style={{ backgroundColor: stem.color }}>
                    {stem.icon}
                  </div>
                  <span className="stem-name">{stem.name}</span>
                </div>

                <div className="channel-controls">
                  {/* Volume Fader */}
                  <div className="fader-container">
                    <div className="fader-track">
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={stem.volume}
                        onChange={(e) => handleVolumeChange(stem.id, e.target.value)}
                        className="fader-input"
                        orient="vertical"
                      />
                      <div 
                        className="fader-fill"
                        style={{ height: `${stem.volume}%` }}
                      />
                    </div>
                    <div className="fader-meter">
                      {[...Array(10)].map((_, i) => (
                        <div 
                          key={i}
                          className={`meter-bar ${i < stem.volume / 10 ? 'active' : ''}`}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Pan Knob */}
                  <div className="pan-control">
                    <div className="pan-knob">
                      <div 
                        className="knob-indicator"
                        style={{ transform: `rotate(${stem.pan * 1.8}deg)` }}
                      />
                    </div>
                    <input
                      type="range"
                      min="-50"
                      max="50"
                      value={stem.pan}
                      onChange={(e) => handlePanChange(stem.id, e.target.value)}
                      className="pan-slider"
                    />
                    <span className="pan-value">{stem.pan > 0 ? 'R' : stem.pan < 0 ? 'L' : 'C'}{Math.abs(stem.pan)}</span>
                  </div>
                </div>

                {/* Channel Buttons */}
                <div className="channel-buttons">
                  <button 
                    className={`channel-btn mute ${stem.muted ? 'active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleMute(stem.id);
                    }}
                  >
                    M
                  </button>
                  <button 
                    className={`channel-btn solo ${stem.solo ? 'active' : ''}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleSolo(stem.id);
                    }}
                  >
                    S
                  </button>
                </div>

                {/* Volume Display */}
                <div className="volume-display">
                  {stem.muted ? (
                    <VolumeX size={16} />
                  ) : (
                    <Volume2 size={16} />
                  )}
                  <span>{stem.volume}%</span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Effects Panel */}
        <div className="effects-panel card-glass">
          <div className="effects-header">
            <Waves size={20} />
            <h3>Effects</h3>
            <button 
              className={`toggle-effects ${showEffects ? 'active' : ''}`}
              onClick={() => setShowEffects(!showEffects)}
            >
              {showEffects ? 'Hide' : 'Show'} Details
            </button>
          </div>

          {selectedStem ? (
            <div className="effects-content">
              {(() => {
                const stem = stems.find(s => s.id === selectedStem);
                if (!stem) return null;
                
                return (
                  <>
                    <div className="effect-section">
                      <h4>{stem.name} - Effects Chain</h4>
                      
                      {/* EQ */}
                      <div className="effect-control">
                        <div className="effect-header">
                          <Filter size={16} />
                          <span>EQ</span>
                          <span className="effect-value">{stem.effects.eq}%</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={stem.effects.eq}
                          onChange={(e) => handleEffectChange(stem.id, 'eq', e.target.value)}
                          className="effect-slider"
                        />
                        <div className="eq-visualization">
                          <div 
                            className="eq-bar"
                            style={{ height: `${stem.effects.eq * 0.8}%` }}
                          />
                          <div 
                            className="eq-bar"
                            style={{ height: `${stem.effects.eq * 0.6}%` }}
                          />
                          <div 
                            className="eq-bar"
                            style={{ height: `${stem.effects.eq * 0.9}%` }}
                          />
                          <div 
                            className="eq-bar"
                            style={{ height: `${stem.effects.eq * 0.7}%` }}
                          />
                        </div>
                      </div>

                      {/* Reverb */}
                      <div className="effect-control">
                        <div className="effect-header">
                          <Waves size={16} />
                          <span>Reverb</span>
                          <span className="effect-value">{stem.effects.reverb}%</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={stem.effects.reverb}
                          onChange={(e) => handleEffectChange(stem.id, 'reverb', e.target.value)}
                          className="effect-slider"
                        />
                      </div>

                      {/* Delay */}
                      <div className="effect-control">
                        <div className="effect-header">
                          <Activity size={16} />
                          <span>Delay</span>
                          <span className="effect-value">{stem.effects.delay}%</span>
                        </div>
                        <input
                          type="range"
                          min="0"
                          max="100"
                          value={stem.effects.delay}
                          onChange={(e) => handleEffectChange(stem.id, 'delay', e.target.value)}
                          className="effect-slider"
                        />
                      </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="quick-actions">
                      <button className="quick-action">
                        <Crosshair size={16} />
                        Auto-Mix
                      </button>
                      <button className="quick-action">
                        <Filter size={16} />
                        Reset EQ
                      </button>
                      <button className="quick-action">
                        <Waves size={16} />
                        Presets
                      </button>
                    </div>
                  </>
                );
              })()}
            </div>
          ) : (
            <div className="no-selection">
              <p>Select a stem to edit effects</p>
            </div>
          )}
        </div>

        {/* Stereo Field */}
        <div className="stereo-field card-glass">
          <div className="field-header">
            <Activity size={20} />
            <h3>Stereo Field</h3>
          </div>

          <div className="field-visualization">
            <div className="field-grid">
              {stems.map((stem) => (
                <motion.div
                  key={stem.id}
                  className="field-dot"
                  style={{ 
                    backgroundColor: stem.color,
                    left: `${50 + stem.pan}%`,
                    bottom: `${stem.volume}%`,
                    opacity: stem.muted ? 0.3 : 1
                  }}
                  animate={{
                    scale: stem.solo ? 1.5 : 1,
                    boxShadow: stem.solo ? `0 0 20px ${stem.color}` : 'none'
                  }}
                  onClick={() => setSelectedStem(stem.id)}
                >
                  <span className="dot-label">{stem.name}</span>
                </motion.div>
              ))}
            </div>
            
            <div className="field-labels">
              <span>L</span>
              <span>C</span>
              <span>R</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Bar - Quick Mix */}
      <div className="quick-mix-bar card-glass">
        <div className="mix-actions">
          <button className="mix-btn">
            <Volume2 size={18} />
            Auto Mix
          </button>
          <button className="mix-btn">
            <Sliders size={18} />
            Reset All
          </button>
          <button className="mix-btn">
            <Crosshair size={18} />
            Center
          </button>
        </div>

        <div className="mix-presets">
          <span className="presets-label">Presets:</span>
          {['Movie Mix', 'Trailer', 'Documentary', 'Cinematic'].map((preset, i) => (
            <button key={i} className="preset-btn">
              {preset}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default StemControl;

