import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users, MessageCircle, UserPlus, Send, Paperclip,
  Smile, MoreVertical, Phone, Video, Settings,
  Clock, Check, CheckCheck, Share2, Copy, Zap
} from 'lucide-react';
import './CollaborativeSession.css';

function CollaborativeSession() {
  const [sessionCode, setSessionCode] = useState('CINE-2024-XK7M');
  const [collaborators, setCollaborators] = useState([
    { id: 1, name: 'You', avatar: '🎵', color: '#d4af37', role: 'admin', online: true },
    { id: 2, name: 'Sarah Chen', avatar: '🎹', color: '#8b5cf6', role: 'editor', online: true },
    { id: 3, name: 'Mike Ross', avatar: '🎸', color: '#06b6d4', role: 'viewer', online: true },
    { id: 4, name: 'Emma Wilson', avatar: '🎻', color: '#10b981', role: 'editor', online: false },
  ]);

  const [messages, setMessages] = useState([
    { id: 1, userId: 2, text: 'Great progress on the battle theme! 🎵', time: '10:30 AM', read: true },
    { id: 2, userId: 1, text: 'Thanks! I think we should add more strings in the climax section.', time: '10:32 AM', read: true },
    { id: 3, userId: 3, text: 'Agreed! I can record some cello samples.', time: '10:35 AM', read: true },
    { id: 4, userId: 4, text: 'The emotion curve looks perfect!', time: '10:40 AM', read: false },
  ]);

  const [newMessage, setNewMessage] = useState('');
  const [isConnected, setIsConnected] = useState(true);
  const [activeTab, setActiveTab] = useState('chat');
  const [showInvite, setShowInvite] = useState(false);
  const [editingTrack, setEditingTrack] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = () => {
    if (newMessage.trim()) {
      setMessages(prev => [...prev, {
        id: Date.now(),
        userId: 1,
        text: newMessage,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        read: false
      }]);
      setNewMessage('');
    }
  };

  const copyInviteLink = () => {
    navigator.clipboard.writeText(`https://cinesonic.ai/join/${sessionCode}`);
  };

  const getCollaboratorById = (id) => collaborators.find(c => c.id === id);

  return (
    <div className="collaborative-session">
      <div className="session-header">
        <div className="session-info">
          <Users className="header-icon" />
          <div>
            <h2>Co-Composer Session</h2>
            <p>Real-time collaboration with AI assistance</p>
          </div>
        </div>

        <div className="session-meta">
          <div className="connection-status">
            <div className={`status-dot ${isConnected ? 'online' : 'offline'}`}></div>
            <span>{isConnected ? 'Connected' : 'Reconnecting...'}</span>
          </div>
          
          <div className="session-code" onClick={() => setShowInvite(true)}>
            <span>Code: {sessionCode}</span>
            <Copy size={14} />
          </div>
        </div>
      </div>

      <div className="session-content">
        <div className="session-main">
          {/* Active Collaborators */}
          <div className="collaborators-bar card-glass">
            <div className="collaborators-list">
              {collaborators.map(user => (
                <div key={user.id} className={`collaborator ${user.online ? 'online' : 'offline'}`}>
                  <div 
                    className="collaborator-avatar"
                    style={{ backgroundColor: user.color }}
                  >
                    {user.avatar}
                    {user.online && <div className="online-indicator"></div>}
                  </div>
                  <div className="collaborator-info">
                    <span className="collaborator-name">{user.name}</span>
                    <span className="collaborator-role">{user.role}</span>
                  </div>
                  {user.id !== 1 && (
                    <button className="more-btn">
                      <MoreVertical size={14} />
                    </button>
                  )}
                </div>
              ))}
            </div>

            <button className="invite-btn" onClick={() => setShowInvite(true)}>
              <UserPlus size={18} />
              Invite
            </button>
          </div>

          {/* Main Workspace */}
          <div className="workspace card-glass">
            <div className="workspace-header">
              <div className="workspace-tabs">
                <button 
                  className={`workspace-tab ${activeTab === 'chat' ? 'active' : ''}`}
                  onClick={() => setActiveTab('chat')}
                >
                  <MessageCircle size={16} />
                  Chat
                </button>
                <button 
                  className={`workspace-tab ${activeTab === 'activity' ? 'active' : ''}`}
                  onClick={() => setActiveTab('activity')}
                >
                  <Clock size={16} />
                  Activity
                </button>
                <button 
                  className={`workspace-tab ${activeTab === 'files' ? 'active' : ''}`}
                  onClick={() => setActiveTab('files')}
                >
                  <Paperclip size={16} />
                  Files
                </button>
              </div>

              <div className="workspace-actions">
                <button className="action-btn">
                  <Settings size={18} />
                </button>
              </div>
            </div>

            <div className="workspace-content">
              {activeTab === 'chat' ? (
                <div className="chat-container">
                  <div className="messages-list">
                    {messages.map(message => {
                      const user = getCollaboratorById(message.userId);
                      const isOwn = message.userId === 1;
                      
                      return (
                        <motion.div
                          key={message.id}
                          className={`message ${isOwn ? 'own' : ''}`}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                        >
                          {!isOwn && (
                            <div 
                              className="message-avatar"
                              style={{ backgroundColor: user.color }}
                            >
                              {user.avatar}
                            </div>
                          )}
                          
                          <div className="message-content">
                            {!isOwn && (
                              <span className="message-name">{user.name}</span>
                            )}
                            <div className="message-bubble">
                              <p>{message.text}</p>
                            </div>
                            <div className="message-meta">
                              <span className="message-time">{message.time}</span>
                              {isOwn && (
                                <span className="message-status">
                                  {message.read ? <CheckCheck size={14} /> : <Check size={14} />}
                                </span>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                    <div ref={messagesEndRef} />
                  </div>

                  <div className="message-input-container">
                    <button className="attach-btn">
                      <Paperclip size={18} />
                    </button>
                    <input
                      type="text"
                      value={newMessage}
                      onChange={(e) => setNewMessage(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                      placeholder="Type a message..."
                      className="message-input"
                    />
                    <button className="emoji-btn">
                      <Smile size={18} />
                    </button>
                    <button 
                      className="send-btn"
                      onClick={sendMessage}
                      disabled={!newMessage.trim()}
                    >
                      <Send size={18} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="activity-container">
                  <div className="activity-list">
                    {[
                      { user: 'Sarah Chen', action: 'edited', target: 'String Arrangement', time: '2 min ago' },
                      { user: 'Mike Ross', action: 'commented on', target: 'Bass Track', time: '5 min ago' },
                      { user: 'Emma Wilson', action: 'exported', target: 'Final Mix', time: '10 min ago' },
                      { user: 'You', action: 'generated', target: '3 new variations', time: '15 min ago' },
                    ].map((activity, i) => (
                      <div key={i} className="activity-item">
                        <div className="activity-icon">
                          <Zap size={14} />
                        </div>
                        <div className="activity-content">
                          <p>
                            <strong>{activity.user}</strong> {activity.action} <span className="highlight">{activity.target}</span>
                          </p>
                          <span className="activity-time">{activity.time}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* AI Suggestions Panel */}
          <div className="ai-suggestions card-glass">
            <div className="suggestions-header">
              <Zap size={20} />
              <h3>AI Co-Composer Suggestions</h3>
            </div>

            <div className="suggestion-cards">
              <div className="suggestion-card">
                <div className="suggestion-icon">🎻</div>
                <div className="suggestion-content">
                  <h4>String Enhancement</h4>
                  <p>Add layered strings to enhance emotional impact in measures 24-32</p>
                </div>
                <button className="apply-btn">Apply</button>
              </div>

              <div className="suggestion-card">
                <div className="suggestion-icon">🎵</div>
                <div className="suggestion-content">
                  <h4>Tempo Adjustment</h4>
                  <p>Gradually increase tempo by 10 BPM for the climax section</p>
                </div>
                <button className="apply-btn">Apply</button>
              </div>

              <div className="suggestion-card">
                <div className="suggestion-icon">🎹</div>
                <div className="suggestion-content">
                  <h4>Harmonic Variation</h4>
                  <p>Try adding a minor key shift for more tension before resolution</p>
                </div>
                <button className="apply-btn">Apply</button>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar - Project Status */}
        <aside className="session-sidebar">
          <div className="sidebar-section card-glass">
            <h4>Project Status</h4>
            
            <div className="project-progress">
              <div className="progress-circle">
                <svg viewBox="0 0 100 100">
                  <circle cx="50" cy="50" r="45" fill="none" stroke="var(--color-bg-tertiary)" strokeWidth="8" />
                  <circle 
                    cx="50" cy="50" r="45" 
                    fill="none" 
                    stroke="url(#gradient)" 
                    strokeWidth="8"
                    strokeLinecap="round"
                    strokeDasharray="283"
                    strokeDashoffset="70"
                  />
                  <defs>
                    <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#d4af37" />
                      <stop offset="100%" stopColor="#8b5cf6" />
                    </linearGradient>
                  </defs>
                </svg>
                <span className="progress-value">75%</span>
              </div>
              <div className="progress-info">
                <h5>Epic Battle Theme</h5>
                <p>Last saved 5 min ago</p>
              </div>
            </div>
          </div>

          <div className="sidebar-section card-glass">
            <h4>Quick Share</h4>
            
            <div className="share-options">
              <button className="share-btn" onClick={copyInviteLink}>
                <Copy size={18} />
                <span>Copy Link</span>
              </button>
              <button className="share-btn">
                <Share2 size={18} />
                <span>Social</span>
              </button>
            </div>

            <div className="collaboration-link">
              <input 
                type="text" 
                value={`https://cinesonic.ai/join/${sessionCode}`}
                readOnly 
                className="link-input"
              />
              <button className="copy-btn" onClick={copyInviteLink}>
                <Copy size={16} />
              </button>
            </div>
          </div>

          <div className="sidebar-section card-glass">
            <h4>Recent Activity</h4>
            
            <div className="activity-timeline">
              <div className="timeline-item">
                <div className="timeline-dot"></div>
                <div className="timeline-content">
                  <p>Sarah added violin track</p>
                  <span>2 minutes ago</span>
                </div>
              </div>
              <div className="timeline-item">
                <div className="timeline-dot"></div>
                <div className="timeline-content">
                  <p>Mike adjusted bass levels</p>
                  <span>8 minutes ago</span>
                </div>
              </div>
              <div className="timeline-item">
                <div className="timeline-dot"></div>
                <div className="timeline-content">
                  <p>Emma commented on mix</p>
                  <span>15 minutes ago</span>
                </div>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* Invite Modal */}
      <AnimatePresence>
        {showInvite && (
          <motion.div 
            className="modal-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowInvite(false)}
          >
            <motion.div 
              className="invite-modal"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <h3>Invite Collaborators</h3>
              
              <div className="invite-link-section">
                <label>Share this link</label>
                <div className="invite-link">
                  <input 
                    type="text" 
                    value={`https://cinesonic.ai/join/${sessionCode}`}
                    readOnly 
                  />
                  <button onClick={copyInviteLink}>
                    <Copy size={18} />
                  </button>
                </div>
              </div>

              <div className="invite-options">
                <label>Or invite via email</label>
                <div className="email-input">
                  <input 
                    type="email" 
                    placeholder="Enter email address"
                  />
                  <button className="btn btn-primary">
                    <Send size={16} />
                    Send Invite
                  </button>
                </div>
              </div>

              <div className="invite-permissions">
                <label>Permissions</label>
                <select className="permission-select">
                  <option value="viewer">Can view only</option>
                  <option value="editor">Can edit</option>
                  <option value="admin">Full access</option>
                </select>
              </div>

              <button className="close-modal-btn" onClick={() => setShowInvite(false)}>
                Done
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default CollaborativeSession;

