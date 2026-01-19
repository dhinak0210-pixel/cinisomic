import { useState, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Box, Torus } from '@react-three/drei';
import { motion } from 'framer-motion';
import { Activity, Music2, Waves, Zap, Settings } from 'lucide-react';
import * as THREE from 'three';
import './Visualizations.css';

function AudioSphere({ intensity }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.005;
      meshRef.current.rotation.y += 0.01;
      
      const scale = 1 + (intensity / 100) * 0.5 + (Math.sin(state.clock.elapsedTime * 2) * 0.1);
      meshRef.current.scale.set(scale, scale, scale);
    }
  });

  return (
    <Sphere 
      ref={meshRef} 
      args={[1, 32, 32]} 
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <meshStandardMaterial 
        color={hovered ? "#d4af37" : "#8b5cf6"}
        metalness={0.8}
        roughness={0.2}
        emissive={hovered ? "#d4af37" : "#8b5cf6"}
        emissiveIntensity={0.2 + (intensity / 100) * 0.5}
      />
    </Sphere>
  );
}

function AudioRing({ intensity, count }) {
  const groupRef = useRef();

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.z += 0.002;
    }
  });

  return (
    <group ref={groupRef}>
      {Array.from({ length: count }).map((_, i) => (
        <mesh key={i} rotation={[Math.PI / 2, 0, (i / count) * Math.PI * 2]}>
          <torusGeometry args={[2 + i * 0.3, 0.02, 8, 32]} />
          <meshStandardMaterial 
            color={i % 2 === 0 ? "#06b6d4" : "#d4af37"}
            emissive={i % 2 === 0 ? "#06b6d4" : "#d4af37"}
            emissiveIntensity={0.3}
            transparent
            opacity={0.6}
          />
        </mesh>
      ))}
    </group>
  );
}

function WaveBox({ intensity }) {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.elapsedTime;
      meshRef.current.rotation.x = time * 0.3;
      meshRef.current.rotation.y = time * 0.2;
      
      meshRef.current.position.y = Math.sin(time) * 0.3;
    }
  });

  return (
    <Box ref={meshRef} args={[1.5, 1.5, 1.5]} position={[2, 0, 0]}>
      <meshStandardMaterial 
        color="#10b981"
        metalness={0.6}
        roughness={0.3}
        emissive="#10b981"
        emissiveIntensity={0.3}
      />
    </Box>
  );
}

function Visualizations() {
  const [visualizerType, setVisualizerType] = useState('sphere');
  const [intensity, setIntensity] = useState(50);
  const [showSettings, setShowSettings] = useState(false);
  const [audioData, setAudioData] = useState(Array(32).fill(0));

  useEffect(() => {
    // Simulate audio data
    const interval = setInterval(() => {
      setAudioData(prev => prev.map(() => Math.random() * intensity));
    }, 100);
    
    return () => clearInterval(interval);
  }, [intensity]);

  const visualizers = [
    { id: 'sphere', name: 'Orb', icon: <Activity size={18} /> },
    { id: 'rings', name: 'Rings', icon: <Waves size={18} /> },
    { id: 'cube', name: 'Cube', icon: <Box size={18} /> },
  ];

  return (
    <div className="visualizations">
      <div className="viz-header">
        <div className="viz-header-content">
          <Zap className="header-icon" />
          <div>
            <h2>Advanced Visualizations</h2>
            <p>Real-time 3D audio visualization and analysis</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="visualizer-selector">
            {visualizers.map(viz => (
              <button
                key={viz.id}
                className={`viz-btn ${visualizerType === viz.id ? 'active' : ''}`}
                onClick={() => setVisualizerType(viz.id)}
              >
                {viz.icon}
                {viz.name}
              </button>
            ))}
          </div>

          <button 
            className={`settings-btn ${showSettings ? 'active' : ''}`}
            onClick={() => setShowSettings(!showSettings)}
          >
            <Settings size={18} />
          </button>
        </div>
      </div>

      <div className="viz-content">
        {/* 3D Canvas */}
        <div className="viz-canvas card-glass">
          <Canvas camera={{ position: [0, 0, 5] }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            <pointLight position={[-10, -10, -10]} color="#8b5cf6" intensity={0.5} />
            
            {visualizerType === 'sphere' && (
              <>
                <AudioSphere intensity={intensity} />
                <AudioRing intensity={intensity} count={8} />
              </>
            )}
            
            {visualizerType === 'rings' && (
              <>
                <AudioRing intensity={intensity} count={12} />
                <AudioRing intensity={intensity * 0.8} count={8} />
              </>
            )}
            
            {visualizerType === 'cube' && (
              <>
                <WaveBox intensity={intensity} />
                <Torus args={[2, 0.05, 16, 100]} rotation={[Math.PI / 2, 0, 0]}>
                  <meshStandardMaterial 
                    color="#f59e0b"
                    emissive="#f59e0b"
                    emissiveIntensity={0.5}
                  />
                </Torus>
              </>
            )}
            
            <OrbitControls enableZoom={true} enablePan={false} />
          </Canvas>
        </div>

        {/* Controls Panel */}
        <div className="viz-controls card-glass">
          <div className="control-section">
            <label>
              <Activity size={16} />
              Intensity
            </label>
            <input
              type="range"
              min="0"
              max="100"
              value={intensity}
              onChange={(e) => setIntensity(e.target.value)}
              className="control-slider"
            />
            <span className="control-value">{intensity}%</span>
          </div>

          <div className="control-section">
            <label>
              <Music2 size={16} />
              Audio Levels
            </label>
            <div className="audio-bars">
              {audioData.map((value, i) => (
                <motion.div
                  key={i}
                  className="audio-bar"
                  initial={{ height: 0 }}
                  animate={{ height: `${value}%` }}
                  transition={{ duration: 0.1 }}
                  style={{
                    background: `linear-gradient(to top, var(--color-gold-primary), var(--color-purple-primary))`
                  }}
                />
              ))}
            </div>
          </div>

          <div className="control-section">
            <label>
              <Waves size={16} />
              Waveform
            </label>
            <div className="waveform-display">
              {audioData.map((value, i) => (
                <div
                  key={i}
                  className="waveform-point"
                  style={{
                    height: `${value / 2}%`,
                    backgroundColor: `hsl(${value * 2}, 70%, 60%)`
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Spectral Analysis */}
        <div className="spectral-analysis card-glass">
          <h3>Spectral Analysis</h3>
          
          <div className="spectrum-bars">
            {Array.from({ length: 32 }).map((_, i) => (
              <div key={i} className="spectrum-bar-container">
                <motion.div
                  className="spectrum-bar"
                  style={{
                    height: `${Math.random() * 100}%`,
                    background: `hsl(${240 - (i * 7)}, 70%, ${50 + (i % 3) * 10}%)`
                  }}
                  animate={{
                    height: [
                      `${Math.random() * 60 + 20}%`,
                      `${Math.random() * 80 + 20}%`,
                      `${Math.random() * 50 + 30}%`
                    ]
                  }}
                  transition={{
                    duration: 0.3 + (i * 0.02),
                    repeat: Infinity,
                    repeatType: "reverse"
                  }}
                />
                <span className="spectrum-label">{i % 4 === 0 ? `${i * 100}Hz` : ''}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Emotion Heatmap */}
        <div className="emotion-heatmap card-glass">
          <h3>Emotion Heatmap</h3>
          
          <div className="heatmap-grid">
            {['Tension', 'Release', 'Intensity', 'Warmth', 'Darkness', 'Energy', 'Hope', 'Dread'].map((emotion, i) => (
              <div key={emotion} className="heatmap-cell">
                <span className="heatmap-label">{emotion}</span>
                <motion.div 
                  className="heatmap-value"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  style={{
                    background: `linear-gradient(90deg, 
                      hsl(${i * 45}, 70%, 50%) 0%, 
                      hsl(${i * 45 + 30}, 70%, 60%) 100%)`
                  }}
                >
                  <motion.div 
                    className="heatmap-progress"
                    initial={{ width: 0 }}
                    animate={{ width: `${30 + Math.random() * 70}%` }}
                    transition={{ duration: 1, delay: i * 0.1 }}
                  />
                </motion.div>
                <span className="heatmap-number">{Math.round(30 + Math.random() * 70)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <motion.div 
          className="settings-panel card-glass"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3>Visualization Settings</h3>
          
          <div className="settings-grid">
            <div className="setting-item">
              <label>Bloom Strength</label>
              <input type="range" min="0" max="100" defaultValue="50" />
            </div>
            
            <div className="setting-item">
              <label>Rotation Speed</label>
              <input type="range" min="0" max="100" defaultValue="50" />
            </div>
            
            <div className="setting-item">
              <label>Particle Count</label>
              <input type="range" min="100" max="1000" defaultValue="500" />
            </div>
            
            <div className="setting-item">
              <label>Color Scheme</label>
              <select>
                <option>Gold & Purple</option>
                <option>Cyan & Blue</option>
                <option>Warm Orange</option>
                <option>Cold Steel</option>
              </select>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default Visualizations;

