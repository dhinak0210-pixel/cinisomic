\\\\\\\\\\\import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Play, Pause, SkipBack, SkipForward, Volume2,
  Clock, Video, Music, Scissors, Layers,
  Maximize2, ZoomIn, ZoomOut, Grid, AlignLeft
} from 'lucide-react';
import './FilmTimeline.css';

function FilmTimeline() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration] = useState(180); // 3 minutes
  const [zoom, setZoom] = useState(100);
  const [showBeats, setShowBeats] = useState(true);
  const [showScenes, setShowScenes] = useState(true);
  const [selectedMarker, setSelectedMarker] = useState(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  
  const timelineRef = useRef(null);
  const playInterval = useRef(null);

  // Scene markers
  const scenes = [
    { id: 1, start: 0, end: 30, name: 'Opening', color: '#8b5cf6', type: 'scene' },
    { id: 2, start: 30, end: 60, name: 'Build-up', color: '#06b6d4', type: 'scene' },
    { id: 3, start: 60, end: 100, name: 'Climax', color: '#f59e0b', type: 'scene' },
    { id: 4, start: 100, end: 130, name: 'Resolution', color: '#10b981', type: 'scene' },
    { id: 5, start: 130, end: 180, name: 'Outro', color: '#8b5cf6', type: 'scene' },
  ];

  // Beat markers
  const beats = [];
  for (let i = 0; i <= duration; i += 2) {
    beats.push({ time: i, major: i % 8 === 0 });
  }

  // Scene markers converted to timeline markers
  const markers = scenes.flatMap(scene => 
    Array.from({ length: Math.floor((scene.end - scene.start) / 10) }, (_, i) => ({
      time: scene.start + i * 10,
      sceneId: scene.id,
      color: scene.color,
      label: `${(scene.start + i * 10)}s`
    }))
  );

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const frames = Math.floor((seconds % 1) * 24);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}:${frames.toString().padStart(2, '0')}`;
  };

  const formatTimeShort = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handlePlayPause = () => {
    if (isPlaying) {
      setIsPlaying(false);
      clearInterval(playInterval.current);
    } else {
      setIsPlaying(true);
      playInterval.current = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= duration) {
            setIsPlaying(false);
            clearInterval(playInterval.current);
            return 0;
          }
          return prev + 1/30;
        });
      }, 1000/30);
    }
  };

  const handleTimelineClick = (e) => {
    if (timelineRef.current) {
      const rect = timelineRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const percentage = x / rect.width;
      setCurrentTime(percentage * duration);
    }
  };

  const handleZoom = (direction) => {
    setZoom(prev => {
      if (direction === 'in') return Math.min(prev + 25, 200);
      if (direction === 'out') return Math.max(prev - 25, 50);
      return prev;
    });
  };

  const getSceneColor = (time) => {
    const scene = scenes.find(s => time >= s.start && time < s.end);
    return scene?.color || '#4a4a5c';
  };

  const getCurrentScene = () => {
    return scenes.find(s => currentTime >= s.start && currentTime < s.end);
  };

  return (
    <div className="film-timeline">
      <div className="timeline-header">
        <div className="timeline-header-content">
          <Video className="header-icon" />
          <div>
            <h2>Film Timeline Sync</h2>
            <p>Frame-accurate synchronization with visual content</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="zoom-controls">
            <button onClick={() => handleZoom('out')} disabled={zoom <= 50}>
              <ZoomOut size={18} />
            </button>
            <span>{zoom}%</span>
            <button onClick={() => handleZoom('in')} disabled={zoom >= 200}>
              <ZoomIn size={18} />
            </button>
          </div>

          <div className="view-toggles">
            <button 
              className={`toggle-btn ${showBeats ? 'active' : ''}`}
              onClick={() => setShowBeats(!showBeats)}
            >
              <Grid size={16} />
              Beats
            </button>
            <button 
              className={`toggle-btn ${showScenes ? 'active' : ''}`}
              onClick={() => setShowScenes(!showScenes)}
            >
              <Layers size={16} />
              Scenes
            </button>
          </div>
        </div>
      </div>

      <div className="timeline-content">
        {/* Main Timeline Area */}
        <div className="timeline-main">
          {/* Video Preview */}
          <div className="video-preview card-glass">
            <div className="video-placeholder">
              {!videoLoaded ? (
                <div className="no-video">
                  <Video size={48} />
                  <p>Load video to preview sync</p>
                  <button className="btn btn-primary">
                    Load Video
                  </button>
                </div>
              ) : (
                <div className="video-active">
                  <div className="video-frame">
                    <span className="frame-time">{formatTime(currentTime)}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Transport Controls */}
            <div className="transport-controls">
              <button className="transport-btn">
                <SkipBack size={20} />
              </button>
              <button 
                className={`transport-btn play ${isPlaying ? 'playing' : ''}`}
                onClick={handlePlayPause}
              >
                {isPlaying ? <Pause size={24} /> : <Play size={24} />}
              </button>
              <button className="transport-btn">
                <SkipForward size={20} />
              </button>
              
              <div className="time-display">
                <Clock size={16} />
                <span>{formatTime(currentTime)}</span>
                <span className="separator">/</span>
                <span>{formatTime(duration)}</span>
              </div>

              <div className="volume-control">
                <Volume2 size={18} />
                <input type="range" min="0" max="100" defaultValue="80" className="volume-slider" />
              </div>
            </div>
          </div>

          {/* Current Scene Info */}
          <div className="scene-info card-glass">
            {getCurrentScene() ? (
              <>
                <div 
                  className="scene-badge"
                  style={{ backgroundColor: getCurrentScene().color }}
                >
                  {getCurrentScene().name}
                </div>
                <div className="scene-timing">
                  <span>{formatTimeShort(getCurrentScene().start)}</span>
                  <div className="scene-progress">
                    <div 
                      className="scene-progress-fill"
                      style={{ 
                        width: `${((currentTime - getCurrentScene().start) / (getCurrentScene().end - getCurrentScene().start)) * 100}%`,
                        backgroundColor: getCurrentScene().color
                      }}
                    />
                  </div>
                  <span>{formatTimeShort(getCurrentScene().end)}</span>
                </div>
              </>
            ) : (
              <span className="no-scene">No scene active</span>
            )}
          </div>

          {/* Timeline */}
          <div className="timeline-container card-glass">
            <div className="timeline-ruler">
              {Array.from({ length: Math.ceil(duration / 10) }, (_, i) => (
                <div key={i} className="ruler-mark">
                  <span className="ruler-label">{formatTimeShort(i * 10)}</span>
                </div>
              ))}
            </div>

            <div 
              className="timeline-track"
              ref={timelineRef}
              onClick={handleTimelineClick}
              style={{ transform: `scaleX(${zoom / 100})`, transformOrigin: 'left' }}
            >
              {/* Scene Backgrounds */}
              {scenes.map(scene => (
                <div
                  key={scene.id}
                  className="scene-segment"
                  style={{
                    left: `${(scene.start / duration) * 100}%`,
                    width: `${((scene.end - scene.start) / duration) * 100}%`,
                    backgroundColor: `${scene.color}20`
                  }}
                >
                  <span className="segment-label">{scene.name}</span>
                </div>
              ))}

              {/* Beat Markers */}
              {showBeats && (
                <div className="beat-markers">
                  {beats.map((beat, i) => (
                    <div
                      key={i}
                      className={`beat-marker ${beat.major ? 'major' : 'minor'}`}
                      style={{ left: `${(beat.time / duration) * 100}%` }}
                    />
                  ))}
                </div>
              )}

              {/* Playhead */}
              <div 
                className="playhead"
                style={{ left: `${(currentTime / duration) * 100}%` }}
              >
                <div className="playhead-head"></div>
                <div className="playhead-line"></div>
              </div>

              {/* Scene Markers */}
              {showScenes && markers.map((marker, i) => (
                <div
                  key={i}
                  className={`scene-marker ${selectedMarker === marker.time ? 'selected' : ''}`}
                  style={{ 
                    left: `${(marker.time / duration) * 100}%`,
                    backgroundColor: marker.color
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    setCurrentTime(marker.time);
                    setSelectedMarker(marker.time);
                  }}
                  title={marker.label}
                />
              ))}
            </div>
          </div>

          {/* Sync Options */}
          <div className="sync-options card-glass">
            <div className="sync-option">
              <AlignLeft size={18} />
              <span>Beat Snap</span>
              <label className="switch">
                <input type="checkbox" defaultChecked />
                <span className="slider"></span>
              </label>
            </div>
            
            <div className="sync-option">
              <Music size={18} />
              <span>Scene Markers</span>
              <label className="switch">
                <input type="checkbox" defaultChecked />
                <span className="slider"></span>
              </label>
            </div>

            <div className="sync-option">
              <Scissors size={18} />
              <span>Auto-Cut</span>
              <label className="switch">
                <input type="checkbox" />
                <span className="slider"></span>
              </label>
            </div>

            <button className="btn btn-primary">
              <Maximize2 size={16} />
              Generate Music
            </button>
          </div>
        </div>

        {/* Sidebar - Scene Markers */}
        <aside className="timeline-sidebar">
          <div className="sidebar-section card-glass">
            <h4>
              <Layers size={18} />
              Scene Markers
            </h4>
            
            <div className="marker-list">
              {scenes.map(scene => (
                <div 
                  key={scene.id} 
                  className="marker-item"
                  onClick={() => {
                    setCurrentTime(scene.start);
                    setSelectedMarker(scene.start);
                  }}
                >
                  <div 
                    className="marker-color"
                    style={{ backgroundColor: scene.color }}
                  />
                  <div className="marker-info">
                    <span className="marker-name">{scene.name}</span>
                    <span className="marker-time">
                      {formatTimeShort(scene.start)} - {formatTimeShort(scene.end)}
                    </span>
                  </div>
                  <button className="marker-action">
                    <Play size={14} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div className="sidebar-section card-glass">
            <h4>
              <Clock size={18} />
              Cue Points
            </h4>
            
            <div className="cue-list">
              {[
                { time: 15, label: 'Title Card', type: 'action' },
                { time: 45, label: 'Character Intro', type: 'dialogue' },
                { time: 75, label: 'First Action', type: 'action' },
                { time: 110, label: 'Climax Start', type: 'climax' },
                { time: 150, label: 'Resolution', type: 'calm' },
              ].map((cue, i) => (
                <div key={i} className="cue-item">
                  <div className="cue-time">{formatTimeShort(cue.time)}</div>
                  <div className="cue-info">
                    <span className="cue-label">{cue.label}</span>
                    <span className="cue-type">{cue.type}</span>
                  </div>
                  <button className="cue-action">
                    +
                  </button>
                </div>
              ))}
            </div>

            <button className="add-cue-btn">
              + Add Cue Point
            </button>
          </div>

          <div className="sidebar-section card-glass">
            <h4>
              <Video size={18} />
              Video Info
            </h4>
            
            <div className="video-info-grid">
              <div className="info-item">
                <span className="info-label">Resolution</span>
                <span className="info-value">1920x1080</span>
              </div>
              <div className="info-item">
                <span className="info-label">Frame Rate</span>
                <span className="info-value">24 fps</span>
              </div>
              <div className="info-item">
                <span className="info-label">Duration</span>
                <span className="info-value">3:00</span>
              </div>
              <div className="info-item">
                <span className="info-label">Format</span>
                <span className="info-value">ProRes</span>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}

export default FilmTimeline;

