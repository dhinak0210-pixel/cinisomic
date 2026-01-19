import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Languages, Mic, Sparkles, Music2, Undo2, Redo2,
  Save, Copy, Download, Wand2, Volume2, Check,
  AlertCircle, Move, GripVertical, Trash2, Plus
} from 'lucide-react';
import './LyricIntelligence.css';

function LyricIntelligence() {
  const [lyrics, setLyrics] = useState([
    { id: 1, text: "In the darkness of the night", translation: "", syllables: 6, rhyme: "night", sentiment: "neutral" },
    { id: 2, text: "A melody takes flight", translation: "", syllables: 6, rhyme: "flight", sentiment: "hopeful" },
    { id: 3, text: "Hearts entwined in eternal light", translation: "", syllables: 7, rhyme: "light", sentiment: "romantic" },
    { id: 4, text: "Love burns bright through the fight", translation: "", syllables: 7, rhyme: "fight", sentiment: "passionate" },
  ]);

  const [selectedLanguage, setSelectedLanguage] = useState('en');
  const [targetLanguage, setTargetLanguage] = useState('es');
  const [isTranslating, setIsTranslating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showTranslation, setShowTranslation] = useState(false);
  const [selectedLines, setSelectedLines] = useState([]);
  const [editMode, setEditMode] = useState('direct');
  const [suggestions, setSuggestions] = useState({});
  const [songStructure, setSongStructure] = useState('verse');

  const languages = [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'es', name: 'Spanish', flag: '🇪🇸' },
    { code: 'fr', name: 'French', flag: '🇫🇷' },
    { code: 'de', name: 'German', flag: '🇩🇪' },
    { code: 'it', name: 'Italian', flag: '🇮🇹' },
    { code: 'pt', name: 'Portuguese', flag: '🇵🇹' },
    { code: 'ja', name: 'Japanese', flag: '🇯🇵' },
    { code: 'ko', name: 'Korean', flag: '🇰🇷' },
    { code: 'zh', name: 'Chinese', flag: '🇨🇳' },
    { code: 'ru', name: 'Russian', flag: '🇷🇺' },
  ];

  const structures = [
    { id: 'intro', name: 'Intro', icon: '🎵' },
    { id: 'verse', name: 'Verse', icon: '📝' },
    { id: 'pre-chorus', name: 'Pre-Chorus', icon: '📈' },
    { id: 'chorus', name: 'Chorus', icon: '🎶' },
    { id: 'bridge', name: 'Bridge', icon: '🌉' },
    { id: 'outro', name: 'Outro', icon: '🎼' },
  ];

  const handleTextChange = (id, text) => {
    setLyrics(prev => prev.map(line => 
      line.id === id ? { ...line, text } : line
    ));
    analyzeLine(id, text);
  };

  const analyzeLine = useCallback((id, text) => {
    // Simulate AI analysis
    const words = text.split(' ').length;
    const syllables = Math.max(1, Math.round(words * 1.5));
    const sentiment = text.toLowerCase().includes('love') ? 'romantic' :
                     text.toLowerCase().includes('dark') ? 'dark' :
                     text.toLowerCase().includes('light') ? 'hopeful' : 'neutral';
    
    setLyrics(prev => prev.map(line =>
      line.id === id ? { ...line, syllables, sentiment } : line
    ));
  }, []);

  const translateLyrics = async () => {
    setIsTranslating(true);
    
    const translations = {
      1: "En la oscuridad de la noche",
      2: "Una melodía emprende el vuelo",
      3: "Corazón entrelazado en luz eterna",
      4: "El amor arde brillante a través de la lucha"
    };

    await new Promise(resolve => setTimeout(resolve, 1500));

    setLyrics(prev => prev.map(line => ({
      ...line,
      translation: translations[line.id] || line.translation
    })));

    setIsTranslating(false);
    setShowTranslation(true);
  };

  const generateRhymes = async (lineId) => {
    setIsAnalyzing(true);
    
    const rhymeSuggestions = {
      1: ['night', 'light', 'bright', 'fight', 'sight', 'delight', 'flight'],
      2: ['flight', 'might', 'right', 'sight', 'height', 'knight', 'bite'],
      3: ['light', 'right', 'bright', 'sight', 'height', 'delight', 'tight'],
      4: ['fight', 'night', 'right', 'bright', 'light', 'tight', 'height']
    };

    await new Promise(resolve => setTimeout(resolve, 800));
    
    setSuggestions(prev => ({
      ...prev,
      [lineId]: rhymeSuggestions[lineId] || []
    }));

    setIsAnalyzing(false);
  };

  const addLine = () => {
    const newId = Math.max(...lyrics.map(l => l.id)) + 1;
    setLyrics(prev => [...prev, {
      id: newId,
      text: '',
      translation: '',
      syllables: 0,
      rhyme: '',
      sentiment: 'neutral'
    }]);
  };

  const deleteLine = (id) => {
    setLyrics(prev => prev.filter(line => line.id !== id));
  };

  const toggleLineSelection = (id) => {
    setSelectedLines(prev => 
      prev.includes(id) ? prev.filter(l => l !== id) : [...prev, id]
    );
  };

  const getSentimentColor = (sentiment) => {
    const colors = {
      romantic: '#f472b6',
      hopeful: '#10b981',
      dark: '#6366f1',
      passionate: '#ef4444',
      neutral: '#6b7280'
    };
    return colors[sentiment] || colors.neutral;
  };

  const getStructureColor = (structure) => {
    const colors = {
      'intro': '#8b5cf6',
      'verse': '#06b6d4',
      'pre-chorus': '#f59e0b',
      'chorus': '#d4af37',
      'bridge': '#ec4899',
      'outro': '#10b981'
    };
    return colors[structure] || colors.verse;
  };

  const copyToClipboard = () => {
    const text = lyrics.map(l => l.text).join('\n');
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="lyric-intelligence">
      <div className="lyric-header">
        <div className="lyric-header-content">
          <Languages className="header-icon" />
          <div>
            <h2>Lyric Intelligence Hub</h2>
            <p>Multilingual lyrics with AI-powered analysis, translation, and suggestions</p>
          </div>
        </div>

        <div className="header-actions">
          <button className="action-btn" onClick={copyToClipboard}>
            <Copy size={18} />
            Copy
          </button>
          <button className="action-btn">
            <Download size={18} />
            Export
          </button>
        </div>
      </div>

      <div className="lyric-content">
        {/* Main Editor */}
        <div className="lyric-main">
          {/* Language & Translation Bar */}
          <div className="language-bar card-glass">
            <div className="language-selector">
              <label>Source Language</label>
              <select 
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="language-select"
              >
                {languages.map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag} {lang.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="translation-controls">
              <button 
                className="translate-btn"
                onClick={translateLyrics}
                disabled={isTranslating}
              >
                {isTranslating ? (
                  <>
                    <div className="spinner-small"></div>
                    Translating...
                  </>
                ) : (
                  <>
                    <Wand2 size={18} />
                    Auto-Translate
                  </>
                )}
              </button>

              <div className="language-selector">
                <label>To</label>
                <select 
                  value={targetLanguage}
                  onChange={(e) => setTargetLanguage(e.target.value)}
                  className="language-select"
                >
                  {languages.map(lang => (
                    <option key={lang.code} value={lang.code}>
                      {lang.flag} {lang.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button 
              className={`toggle-translation ${showTranslation ? 'active' : ''}`}
              onClick={() => setShowTranslation(!showTranslation)}
            >
              {showTranslation ? 'Hide' : 'Show'} Translation
            </button>
          </div>

          {/* Song Structure */}
          <div className="structure-bar card-glass">
            <div className="structure-label">
              <Music2 size={16} />
              Song Structure
            </div>
            <div className="structure-tabs">
              {structures.map(struct => (
                <button
                  key={struct.id}
                  className={`structure-tab ${songStructure === struct.id ? 'active' : ''}`}
                  style={{
                    '--tab-color': getStructureColor(struct.id)
                  }}
                  onClick={() => setSongStructure(struct.id)}
                >
                  <span className="structure-icon">{struct.icon}</span>
                  {struct.name}
                </button>
              ))}
            </div>
          </div>

          {/* Lyrics Editor */}
          <div className="lyrics-editor card-glass">
            <div className="editor-header">
              <div className="editor-title">
                <Sparkles size={20} />
                <h3>Lyrics Editor</h3>
              </div>
              <div className="editor-actions">
                <button className="editor-btn" title="Undo">
                  <Undo2 size={16} />
                </button>
                <button className="editor-btn" title="Redo">
                  <Redo2 size={16} />
                </button>
                <button className="editor-btn" onClick={addLine}>
                  <Plus size={16} />
                  Add Line
                </button>
              </div>
            </div>

            <div className="lyrics-lines">
              <AnimatePresence>
                {lyrics.map((line, index) => (
                  <motion.div
                    key={line.id}
                    className={`lyric-line ${selectedLines.includes(line.id) ? 'selected' : ''}`}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ delay: index * 0.05 }}
                  >
                    <div className="line-number">{index + 1}</div>
                    
                    <div className="line-drag">
                      <GripVertical size={16} />
                    </div>

                    <div className="line-content">
                      {editMode === 'direct' ? (
                        <input
                          type="text"
                          value={line.text}
                          onChange={(e) => handleTextChange(line.id, e.target.value)}
                          placeholder="Enter lyrics..."
                          className="lyric-input"
                        />
                      ) : (
                        <textarea
                          value={line.text}
                          onChange={(e) => handleTextChange(line.id, e.target.value)}
                          placeholder="Enter lyrics..."
                          className="lyric-textarea"
                          rows={2}
                        />
                      )}

                      {showTranslation && (
                        <div className="line-translation">
                          <Languages size={14} />
                          <span>{line.translation || 'Translation will appear here...'}</span>
                        </div>
                      )}
                    </div>

                    <div className="line-meta">
                      <div 
                        className="sentiment-badge"
                        style={{ backgroundColor: getSentimentColor(line.sentiment) }}
                        title={`Sentiment: ${line.sentiment}`}
                      >
                        {line.sentiment.charAt(0).toUpperCase()}
                      </div>
                      
                      <div className="syllable-count" title={`${line.syllables} syllables`}>
                        {line.syllables} <span className="syllable-label">syll</span>
                      </div>

                      <button 
                        className="line-action"
                        onClick={() => generateRhymes(line.id)}
                        title="Get rhyme suggestions"
                      >
                        <Sparkles size={14} />
                      </button>

                      <button 
                        className="line-action"
                        onClick={() => toggleLineSelection(line.id)}
                        title="Select line"
                      >
                        <Check size={14} />
                      </button>

                      <button 
                        className="line-action delete"
                        onClick={() => deleteLine(line.id)}
                        title="Delete line"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>

                    {/* Rhyme Suggestions */}
                    {suggestions[line.id] && (
                      <motion.div 
                        className="rhyme-suggestions"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                      >
                        <div className="suggestions-header">
                          <Wand2 size={14} />
                          Rhyme Suggestions
                        </div>
                        <div className="suggestions-list">
                          {suggestions[line.id].map((rhyme, i) => (
                            <button
                              key={i}
                              className="suggestion-chip"
                              onClick={() => {
                                handleTextChange(line.id, `${line.text} ${rhyme}`);
                                setSuggestions(prev => ({ ...prev, [line.id]: [] }));
                              }}
                            >
                              {rhyme}
                            </button>
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>

            {/* Add Line Button at Bottom */}
            <button className="add-line-btn" onClick={addLine}>
              <Plus size={20} />
              Add New Line
            </button>
          </div>

          {/* AI Suggestions Panel */}
          <div className="ai-suggestions card-glass">
            <div className="suggestions-title">
              <Sparkles size={20} />
              <h3>AI Songwriting Assistant</h3>
            </div>

            <div className="suggestion-categories">
              <button className="suggestion-category">
                <Wand2 size={20} />
                <span>Continue Lyrics</span>
              </button>
              <button className="suggestion-category">
                <Music2 size={20} />
                <span>Melody Suggestion</span>
              </button>
              <button className="suggestion-category">
                <Languages size={20} />
                <span>Improve Flow</span>
              </button>
              <button className="suggestion-category">
                <Volume2 size={20} />
                <span> pronunciation</span>
              </button>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <aside className="lyric-sidebar">
          {/* Stats */}
          <div className="sidebar-section card-glass">
            <h4>Lyrics Stats</h4>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-value">{lyrics.length}</span>
                <span className="stat-label">Lines</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {lyrics.reduce((acc, l) => acc + l.syllables, 0)}
                </span>
                <span className="stat-label">Syllables</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">
                  {lyrics.filter(l => l.text.split(' ').length).length}
                </span>
                <span className="stat-label">Words</span>
              </div>
              <div className="stat-item">
                <span className="stat-value">4:30</span>
                <span className="stat-label">Est. Duration</span>
              </div>
            </div>
          </div>

          {/* Structure Settings */}
          <div className="sidebar-section card-glass">
            <h4>Structure Settings</h4>
            <div className="setting-group">
              <label>Bars per Line</label>
              <select className="setting-select">
                <option value="2">2 bars</option>
                <option value="4" selected>4 bars</option>
                <option value="8">8 bars</option>
              </select>
            </div>
            <div className="setting-group">
              <label>Rhythm Pattern</label>
              <select className="setting-select">
                <option value="straight">Straight</option>
                <option value="swing">Swing</option>
                <option value="halftime">Half Time</option>
                <option value="double">Double Time</option>
              </select>
            </div>
          </div>

          {/* Voice Selection */}
          <div className="sidebar-section card-glass">
            <h4>Vocal Settings</h4>
            <div className="setting-group">
              <label>Voice</label>
              <select className="setting-select">
                <option value="default">Default AI Voice</option>
                <option value="male1">Professional Male 1</option>
                <option value="male2">Professional Male 2</option>
                <option value="female1">Professional Female 1</option>
                <option value="female2">Professional Female 2</option>
                <option value="clone">Clone My Voice</option>
              </select>
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
            <div className="setting-group">
              <label>Range</label>
              <select className="setting-select">
                <option value="alto">Alto / Baritone</option>
                <option value="soprano">Soprano / Tenor</option>
                <option value="mezzo">Mezzo / Bass-Baritone</option>
              </select>
            </div>
          </div>

          {/* Generate Button */}
          <button className="btn btn-primary btn-block">
            <Music2 size={18} />
            Generate Song from Lyrics
          </button>
        </aside>
      </div>
    </div>
  );
}

export default LyricIntelligence;

