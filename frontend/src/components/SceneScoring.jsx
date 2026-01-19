import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, Film, Play, Pause, Clock, Zap, 
  ChevronRight, Image, Scissors, Music,
  Maximize2, Volume2
} from 'lucide-react';
import './SceneScoring.css';

function SceneScoring() {
  const [videoFile, setVideoFile] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [selectedScene, setSelectedScene] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [sceneDescriptions, setSceneDescriptions] = useState({});
  const [musicSuggestions, setMusicSuggestions] = useState({});
  const videoRef = useRef(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoPreview(url);
      analyzeVideo(file);
    }
  }, []);

  const analyzeVideo = async (file) => {
    setIsAnalyzing(true);
    setAnalysisProgress(0);
    
    // Simulate AI video analysis
    const totalSteps = 5;
    for (let i = 1; i <= totalSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, 800));
      setAnalysisProgress((i / totalSteps) * 100);
    }

    // Simulated scene detection
    const mockScenes = [
      { id: 1, start: 0, end: 15, type: 'opening', confidence: 0.95, thumbnail: null },
      { id: 2, start: 15, end: 35, type: 'action', confidence: 0.88, thumbnail: null },
      { id: 3, start: 35, end: 50, type: 'dialogue', confidence: 0.92, thumbnail: null },
      { id: 4, start: 50, end: 70, type: 'climax', confidence: 0.97, thumbnail: null },
      { id: 5, start: 70, end: 90, type: 'resolution', confidence: 0.91, thumbnail: null },
    ];

    setScenes(mockScenes);
    setIsAnalyzing(false);
  };

  const handleSceneDescription = (sceneId, description) => {
    setSceneDescriptions(prev => ({
      ...prev,
      [sceneId]: description
    }));
    
    // Generate music suggestions based on description
    generateMusicSuggestions(sceneId, description);
  };

  const generateMusicSuggestions = (sceneId, description) => {
    const suggestions = {
      1: { mood: 'mysterious', tempo: 80, instruments: ['strings', 'piano'] },
      2: { mood: 'epic', tempo: 140, instruments: ['orchestra', 'choir', 'drums'] },
      3: { mood: 'emotional', tempo: 70, instruments: ['cello', 'violin'] },
      4: { mood: 'intense', tempo: 160, instruments: ['full-orchestra', 'brass'] },
      5: { mood: 'hopeful', tempo: 90, instruments: ['strings', 'woodwinds'] },
    };
    
    setMusicSuggestions(prev => ({
      ...prev,
      [sceneId]: suggestions[sceneId]
    }));
  };

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
      setDuration(videoRef.current.duration);
    }
  };

  const jumpToScene = (scene) => {
    if (videoRef.current) {
      videoRef.current.currentTime = scene.start;
      setSelectedScene(scene);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getSceneIcon = (type) => {
    switch (type) {
      case 'opening': return '🎬';
      case 'action': return '💥';
      case 'dialogue': return '💬';
      case 'climax': return '⚡';
      case 'resolution': return '🌅';
      default: return '📍';
    }
  };

  const getSceneColor = (type) => {
    switch (type) {
      case 'opening': return '#8b5cf6';
      case 'action': return '#ef4444';
      case 'dialogue': return '#06b6d4';
      case 'climax': return '#f59e0b';
      case 'resolution': return '#10b981';
      default: return '#6b7280';
    }
  };

  return (
    <div className="scene-scoring">
      <div className="scene-scoring-header">
        <div className="scene-header-content">
          <Film className="header-icon" />
          <div>
            <h2>Scene-Aware Scoring</h2>
            <p>Upload your video for AI-powered scene analysis and adaptive music generation</p>
          </div>
        </div>
      </div>

      <div className="scene-scoring-content">
        <div className="scene-main-panel">
          {/* Video Preview Area */}
          <div className="video-preview-area card-glass">
            {!videoPreview ? (
              <div 
                className="upload-zone"
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="upload-content">
                  <div className="upload-icon">
                    <Upload size={48} />
                  </div>
                  <h3>Drop your video here</h3>
                  <p>or click to browse</p>
                  <span className="upload-hint">Supports MP4, MOV, AVI (max 500MB)</span>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileSelect}
                  className="file-input"
                />
              </div>
            ) : (
              <div className="video-player-wrapper">
                <video
                  ref={videoRef}
                  src={videoPreview}
                  className="video-player"
                  onTimeUpdate={handleTimeUpdate}
                  onLoadedMetadata={() => {
                    setDuration(videoRef.current?.duration || 0);
                  }}
                />
                
                {/* Video Overlay Controls */}
                <div className="video-controls">
                  <button className="play-control" onClick={handlePlayPause}>
                    {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                  </button>
                  
                  <div className="timeline-slider">
                    <input
                      type="range"
                      min="0"
                      max={duration || 100}
                      value={currentTime}
                      onChange={(e) => {
                        const time = parseFloat(e.target.value);
                        videoRef.current.currentTime = time;
                        setCurrentTime(time);
                      }}
                      className="video-slider"
                    />
                    <div 
                      className="timeline-progress"
                      style={{ width: `${(currentTime / duration) * 100}%` }}
                    />
                  </div>
                  
                  <div className="time-display">
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </div>
                </div>

                {/* Scene Markers on Timeline */}
                <div className="scene-markers">
                  {scenes.map(scene => (
                    <div
                      key={scene.id}
                      className={`scene-marker ${selectedScene?.id === scene.id ? 'active' : ''}`}
                      style={{
                        left: `${(scene.start / duration) * 100}%`,
                        width: `${((scene.end - scene.start) / duration) * 100}%`,
                        backgroundColor: getSceneColor(scene.type)
                      }}
                      onClick={() => jumpToScene(scene)}
                      title={`${scene.type} (${formatTime(scene.start)} - ${formatTime(scene.end)})`}
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* AI Analysis Progress */}
          <AnimatePresence>
            {isAnalyzing && (
              <motion.div 
                className="analysis-progress card-glass"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <div className="analysis-header">
                  <Zap className="analysis-icon" />
                  <h3>AI Scene Analysis</h3>
                </div>
                <div className="progress-bar-container">
                  <div 
                    className="progress-bar-fill"
                    style={{ width: `${analysisProgress}%` }}
                  />
                </div>
                <p className="progress-status">
                  {analysisProgress < 20 && 'Detecting scene transitions...'}
                  {analysisProgress >= 20 && analysisProgress < 40 && 'Analyzing visual content...'}
                  {analysisProgress >= 40 && analysisProgress < 60 && 'Identifying emotions...'}
                  {analysisProgress >= 60 && analysisProgress < 80 && 'Classifying scene types...'}
                  {analysisProgress >= 80 && 'Generating music recommendations...'}
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Scene Timeline */}
          {scenes.length > 0 && (
            <div className="scene-timeline card-glass">
              <div className="timeline-header">
                <Scissors size={20} />
                <h3>Detected Scenes</h3>
                <span className="scene-count">{scenes.length} scenes found</span>
              </div>
              
              <div className="scenes-grid">
                {scenes.map(scene => (
                  <motion.div
                    key={scene.id}
                    className={`scene-card ${selectedScene?.id === scene.id ? 'selected' : ''}`}
                    style={{ borderColor: getSceneColor(scene.type) }}
                    onClick={() => setSelectedScene(scene)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div 
                      className="scene-type-badge"
                      style={{ backgroundColor: getSceneColor(scene.type) }}
                    >
                      {getSceneIcon(scene.type)} {scene.type}
                    </div>
                    
                    <div className="scene-time">
                      <Clock size={14} />
                      {formatTime(scene.start)} - {formatTime(scene.end)}
                    </div>
                    
                    <div className="scene-confidence">
                      <span>Confidence:</span>
                      <div className="confidence-bar">
                        <div 
                          className="confidence-fill"
                          style={{ width: `${scene.confidence * 100}%` }}
                        />
                      </div>
                      <span className="confidence-value">{(scene.confidence * 100).toFixed(0)}%</span>
                    </div>

                    {/* Scene Description Input */}
                    <div className="scene-description">
                      <Image size={14} />
                      <input
                        type="text"
                        placeholder="Describe this scene..."
                        value={sceneDescriptions[scene.id] || ''}
                        onChange={(e) => handleSceneDescription(scene.id, e.target.value)}
                        className="description-input"
                      />
                    </div>

                    {/* Music Suggestions */}
                    {musicSuggestions[scene.id] && (
                      <div className="music-suggestions">
                        <Music size={14} />
                        <div className="suggestion-tags">
                          <span className="tag mood">{musicSuggestions[scene.id].mood}</span>
                          <span className="tag tempo">{musicSuggestions[scene.id].tempo} BPM</span>
                          {musicSuggestions[scene.id].instruments.map(inst => (
                            <span key={inst} className="tag instrument">{inst}</span>
                          ))}
                        </div>
                      </div>
                    )}

                    <button 
                      className="generate-scene-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        // Generate music for this scene
                      }}
                    >
                      <Music size={16} />
                      Generate Music
                    </button>
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar - Scene Details */}
        {selectedScene && (
          <motion.aside 
            className="scene-sidebar"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="sidebar-header">
              <h3>Scene {selectedScene.id}</h3>
              <span 
                className="scene-type-label"
                style={{ backgroundColor: getSceneColor(selectedScene.type) }}
              >
                {selectedScene.type}
              </span>
            </div>

            <div className="scene-details">
              <div className="detail-row">
                <span className="detail-label">Start Time</span>
                <span className="detail-value">{formatTime(selectedScene.start)}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">End Time</span>
                <span className="detail-value">{formatTime(selectedScene.end)}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Duration</span>
                <span className="detail-value">{formatTime(selectedScene.end - selectedScene.start)}s</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Confidence</span>
                <span className="detail-value">{(selectedScene.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>

            {/* Custom Scene Description */}
            <div className="custom-description">
              <h4>Scene Description</h4>
              <textarea
                placeholder="Add more details about this scene..."
                value={sceneDescriptions[selectedScene.id] || ''}
                onChange={(e) => handleSceneDescription(selectedScene.id, e.target.value)}
                className="description-textarea"
                rows={4}
              />
            </div>

            {/* Music Generation Settings */}
            <div className="generation-settings">
              <h4>Music Settings</h4>
              
              <div className="setting-group">
                <label>Mood</label>
                <select className="setting-select">
                  <option value="epic">Epic</option>
                  <option value="dramatic">Dramatic</option>
                  <option value="mysterious">Mysterious</option>
                  <option value="romantic">Romantic</option>
                  <option value="tense">Tense</option>
                  <option value="joyful">Joyful</option>
                  <option value="melancholic">Melancholic</option>
                </select>
              </div>

              <div className="setting-group">
                <label>Tempo (BPM)</label>
                <input 
                  type="range" 
                  min="60" 
                  max="180" 
                  defaultValue="120"
                  className="setting-slider"
                />
                <span className="slider-value">120 BPM</span>
              </div>

              <div className="setting-group">
                <label>Intensity</label>
                <div className="intensity-buttons">
                  {[1, 2, 3, 4, 5].map(level => (
                    <button key={level} className="intensity-btn">{level}</button>
                  ))}
                </div>
              </div>

              <div className="setting-group">
                <label>Key</label>
                <select className="setting-select">
                  <option value="c">C Major</option>
                  <option value="am">A Minor</option>
                  <option value="g">G Major</option>
                  <option value="em">E Minor</option>
                  <option value="d">D Major</option>
                  <option value="bm">B Minor</option>
                </select>
              </div>
            </div>

            <button className="btn btn-primary btn-block">
              <Zap size={18} />
              Generate Scene Music
            </button>
          </motion.aside>
        )}
      </div>
    </div>
  );
}

export default SceneScoring;

