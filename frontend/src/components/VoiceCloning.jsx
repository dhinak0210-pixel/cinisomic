import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Mic, Upload, Play, Pause, Trash2, Plus,
  Shield, Check, AlertCircle, Clock, User,
  Music2, Volume2, Download, Settings,
  FileAudio, Ear, Zap
} from 'lucide-react';
import './VoiceCloning.css';

function VoiceCloning() {
  const [voiceBank, setVoiceBank] = useState([
    {
      id: 1,
      name: 'My Voice Clone',
      status: 'ready',
      samples: 5,
      duration: '2:34',
      created: '2024-01-15',
      consent: true,
      quality: 92
    },
    {
      id: 2,
      name: 'Narrator Voice',
      status: 'training',
      samples: 3,
      duration: '1:45',
      created: '2024-01-18',
      consent: true,
      quality: null
    }
  ]);

  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [samples, setSamples] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [showConsent, setShowConsent] = useState(false);
  const [consentGiven, setConsentGiven] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [playingSample, setPlayingSample] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        const url = URL.createObjectURL(blob);
        setSamples(prev => [...prev, {
          id: Date.now(),
          url,
          duration: recordingTime
        }]);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
      setRecordingTime(0);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const deleteSample = (id) => {
    setSamples(prev => prev.filter(s => s.id !== id));
  };

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    files.forEach(file => {
      if (file.type.startsWith('audio/')) {
        const url = URL.createObjectURL(file);
        setSamples(prev => [...prev, {
          id: Date.now() + Math.random(),
          url,
          name: file.name,
          duration: 0
        }]);
      }
    });
  };

  const trainVoice = () => {
    if (samples.length < 3) return;
    
    setIsTraining(true);
    setTrainingProgress(0);
    
    const interval = setInterval(() => {
      setTrainingProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          
          // Add new voice to bank
          const newVoice = {
            id: Date.now(),
            name: `Voice ${voiceBank.length + 1}`,
            status: 'ready',
            samples: samples.length,
            duration: formatTime(samples.reduce((acc, s) => acc + s.duration, 0)),
            created: new Date().toISOString().split('T')[0],
            consent: consentGiven,
            quality: 85 + Math.random() * 10
          };
          
          setVoiceBank(prev => [...prev, newVoice]);
          setSamples([]);
          setConsentGiven(false);
          
          return 100;
        }
        return prev + 2;
      });
    }, 100);
  };

  const deleteVoice = (id) => {
    setVoiceBank(prev => prev.filter(v => v.id !== id));
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready': return 'var(--color-success)';
      case 'training': return 'var(--color-warning)';
      case 'error': return 'var(--color-error)';
      default: return 'var(--color-text-muted)';
    }
  };

  return (
    <div className="voice-cloning">
      <div className="voice-header">
        <div className="voice-header-content">
          <User className="header-icon" />
          <div>
            <h2>Voice Cloning Studio</h2>
            <p>Create custom AI voices with full consent and control</p>
          </div>
        </div>
      </div>

      <div className="voice-content">
        <div className="voice-main">
          {/* Recording Studio */}
          <div className="recording-studio card-glass">
            <div className="studio-header">
              <Mic size={20} />
              <h3>Recording Studio</h3>
            </div>

            {/* Recording Controls */}
            <div className="recording-controls">
              <div className="timer-display">
                <Clock size={24} />
                <span className="time">{formatTime(recordingTime)}</span>
              </div>

              <div className="control-buttons">
                {!isRecording ? (
                  <button 
                    className="record-btn"
                    onClick={startRecording}
                  >
                    <Mic size={24} />
                    Start Recording
                  </button>
                ) : (
                  <button 
                    className="record-btn recording"
                    onClick={stopRecording}
                  >
                    <div className="recording-pulse"></div>
                    Stop Recording
                  </button>
                )}

                <label className="upload-btn">
                  <Upload size={18} />
                  Upload Audio
                  <input
                    type="file"
                    accept="audio/*"
                    multiple
                    onChange={handleFileUpload}
                    hidden
                  />
                </label>
              </div>

              <p className="recording-hint">
                Record at least 3 samples (30 seconds each) for best results
              </p>
            </div>

            {/* Samples List */}
            {samples.length > 0 && (
              <div className="samples-section">
                <div className="samples-header">
                  <h4>Voice Samples ({samples.length})</h4>
                  <button 
                    className="clear-btn"
                    onClick={() => setSamples([])}
                  >
                    Clear All
                  </button>
                </div>

                <div className="samples-grid">
                  {samples.map((sample, index) => (
                    <motion.div
                      key={sample.id}
                      className="sample-card"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                    >
                      <div className="sample-waveform">
                        {[...Array(40)].map((_, i) => (
                          <div
                            key={i}
                            className="waveform-bar"
                            style={{
                              height: `${Math.random() * 100}%`,
                              animationDelay: `${i * 0.02}s`
                            }}
                          />
                        ))}
                      </div>

                      <div className="sample-info">
                        <span>Sample {index + 1}</span>
                        {sample.duration > 0 && (
                          <span>{formatTime(sample.duration)}</span>
                        )}
                      </div>

                      <div className="sample-actions">
                        <button 
                          className="play-btn"
                          onClick={() => setPlayingSample(playingSample === sample.id ? null : sample.id)}
                        >
                          {playingSample === sample.id ? <Pause size={16} /> : <Play size={16} />}
                        </button>
                        <button 
                          className="delete-btn"
                          onClick={() => deleteSample(sample.id)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {/* Consent Section */}
            <div className="consent-section">
              <div className="consent-header">
                <Shield size={20} />
                <h4>Voice Consent & Agreement</h4>
              </div>

              <div className="consent-content">
                <div className="consent-item">
                  <Check size={16} />
                  <span>I own the voice being recorded</span>
                </div>
                <div className="consent-item">
                  <Check size={16} />
                  <span>I consent to AI training use</span>
                </div>
                <div className="consent-item">
                  <Check size={16} />
                  <span>I understand voice will be stored securely</span>
                </div>
                <div className="consent-item">
                  <Check size={16} />
                  <span>I can delete this voice at any time</span>
                </div>
              </div>

              <label className="consent-checkbox">
                <input
                  type="checkbox"
                  checked={consentGiven}
                  onChange={(e) => setConsentGiven(e.target.checked)}
                />
                <span className="checkmark"></span>
                <span>I agree to the voice cloning terms and conditions</span>
              </label>
            </div>

            {/* Train Button */}
            <button
              className={`train-btn ${samples.length < 3 || !consentGiven ? 'disabled' : ''}`}
              onClick={trainVoice}
              disabled={samples.length < 3 || !consentGiven || isTraining}
            >
              {isTraining ? (
                <>
                  <div className="spinner-small"></div>
                  Training Voice Model... {trainingProgress}%
                </>
              ) : (
                <>
                  <Zap size={18} />
                  Train Voice Model ({samples.length}/3 min samples)
                </>
              )}
            </button>

            {/* Training Progress */}
            <AnimatePresence>
              {isTraining && (
                <motion.div
                  className="training-progress"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                >
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${trainingProgress}%` }}
                    />
                  </div>
                  <div className="progress-steps">
                    <div className={`step ${trainingProgress > 20 ? 'complete' : ''}`}>
                      <div className="step-icon">1</div>
                      <span>Preprocessing</span>
                    </div>
                    <div className={`step ${trainingProgress > 40 ? 'complete' : ''}`}>
                      <div className="step-icon">2</div>
                      <span>Feature Extraction</span>
                    </div>
                    <div className={`step ${trainingProgress > 70 ? 'complete' : ''}`}>
                      <div className="step-icon">3</div>
                      <span>Model Training</span>
                    </div>
                    <div className={`step ${trainingProgress > 90 ? 'complete' : ''}`}>
                      <div className="step-icon">4</div>
                      <span>Finalizing</span>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Voice Settings */}
          <div className="voice-settings card-glass">
            <div className="settings-header">
              <Settings size={20} />
              <h3>Voice Settings</h3>
            </div>

            <div className="setting-group">
              <label>Voice Style</label>
              <select className="setting-select">
                <option value="natural">Natural</option>
                <option value="expressive">Expressive</option>
                <option value="cinematic">Cinematic</option>
                <option value="soft">Soft & Gentle</option>
                <option value="powerful">Powerful</option>
              </select>
            </div>

            <div className="setting-group">
              <label>Pitch Adjustment</label>
              <input 
                type="range" 
                min="-12" 
                max="12" 
                defaultValue="0"
                className="setting-slider"
              />
              <span className="slider-label">0 semitones</span>
            </div>

            <div className="setting-group">
              <label>Speed</label>
              <input 
                type="range" 
                min="50" 
                max="150" 
                defaultValue="100"
                className="setting-slider"
              />
              <span className="slider-label">100%</span>
            </div>

            <div className="setting-group">
              <label>Emotion Intensity</label>
              <input 
                type="range" 
                min="0" 
                max="100" 
                defaultValue="50"
                className="setting-slider"
              />
              <span className="slider-label">50%</span>
            </div>

            <div className="setting-group">
              <label>Breath Control</label>
              <select className="setting-select">
                <option value="none">No Breath Sounds</option>
                <option value="minimal">Minimal</option>
                <option value="natural">Natural</option>
                <option value="heavy">Heavy Breathing</option>
              </select>
            </div>
          </div>
        </div>

        {/* Sidebar - Voice Bank */}
        <aside className="voice-sidebar">
          <div className="sidebar-header">
            <Ear size={20} />
            <h3>Voice Bank</h3>
          </div>

          <div className="voice-list">
            {voiceBank.map(voice => (
              <motion.div
                key={voice.id}
                className={`voice-card ${selectedVoice === voice.id ? 'selected' : ''}`}
                onClick={() => setSelectedVoice(voice.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="voice-avatar">
                  <User size={24} />
                </div>

                <div className="voice-info">
                  <h4>{voice.name}</h4>
                  <div className="voice-meta">
                    <span 
                      className="voice-status"
                      style={{ color: getStatusColor(voice.status) }}
                    >
                      {voice.status === 'ready' && <Check size={12} />}
                      {voice.status === 'training' && <Clock size={12} />}
                      {voice.status}
                    </span>
                    <span>{voice.samples} samples</span>
                  </div>
                  {voice.quality && (
                    <div className="voice-quality">
                      <div className="quality-bar">
                        <div 
                          className="quality-fill"
                          style={{ width: `${voice.quality}%` }}
                        />
                      </div>
                      <span>{voice.quality.toFixed(0)}%</span>
                    </div>
                  )}
                </div>

                <button 
                  className="voice-delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteVoice(voice.id);
                  }}
                >
                  <Trash2 size={14} />
                </button>
              </motion.div>
            ))}

            {/* Add New Voice */}
            <button className="add-voice-btn">
              <Plus size={24} />
              <span>Add New Voice</span>
            </button>
          </div>

          {/* Selected Voice Details */}
          {selectedVoice && (
            <motion.div 
              className="voice-details card-glass"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h4>Voice Details</h4>
              
              <div className="detail-row">
                <span>Total Duration</span>
                <span>{voiceBank.find(v => v.id === selectedVoice)?.duration}</span>
              </div>
              <div className="detail-row">
                <span>Created</span>
                <span>{voiceBank.find(v => v.id === selectedVoice)?.created}</span>
              </div>
              <div className="detail-row">
                <span>Consent</span>
                <span className="consent-badge">
                  <Shield size={12} />
                  Verified
                </span>
              </div>

              <div className="voice-actions">
                <button className="btn btn-primary btn-block">
                  <Volume2 size={16} />
                  Test Voice
                </button>
                <button className="btn btn-outline btn-block">
                  <Download size={16} />
                  Export Voice
                </button>
              </div>
            </motion.div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default VoiceCloning;

