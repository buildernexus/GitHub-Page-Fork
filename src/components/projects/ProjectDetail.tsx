// src/components/projects/ProjectDetail.tsx
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ArrowLeft, Github, Play, Download,
  Calendar, Clock, CheckCircle, Activity, Star,
  Zap, TrendingUp,
  ChevronRight, Maximize2, X,
  Layers, Shield, Database, Cpu, Globe
} from 'lucide-react';
import { type Project, getProjectById } from '../../data/projectsData';
import './ProjectDetail.css';

interface ProjectDetailProps {
  projectId: string;
  onBack?: () => void;
}

const ProjectDetail: React.FC<ProjectDetailProps> = ({ projectId, onBack }) => {
  const [project, setProject] = useState<Project | undefined>();
  const [activeSection, setActiveSection] = useState('overview');
  const [fullscreenImage, setFullscreenImage] = useState<string | null>(null);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  useEffect(() => {
    const projectData = getProjectById(projectId);
    setProject(projectData);
    window.scrollTo(0, 0);
  }, [projectId]);

  if (!project) {
    return (
      <div className="project-detail-loading">
        <div className="loading-spinner" />
        <p>Loading project...</p>
      </div>
    );
  }

  const sections = [
    { id: 'overview', label: 'Overview', icon: <Globe /> },
    { id: 'tech', label: 'Tech Stack', icon: <Layers /> },
    { id: 'features', label: 'Features', icon: <Zap /> },
    { id: 'challenges', label: 'Challenges', icon: <Shield /> },
    { id: 'impact', label: 'Impact', icon: <TrendingUp /> }
  ];

  const techCategories = {
    frontend: { label: 'Frontend', icon: <Globe />, color: '#667eea' },
    backend: { label: 'Backend', icon: <Database />, color: '#00d4aa' },
    database: { label: 'Database', icon: <Database />, color: '#ffd700' },
    ml: { label: 'Machine Learning', icon: <Cpu />, color: '#f093fb' },
    cloud: { label: 'Cloud & DevOps', icon: <Globe />, color: '#ff6b6b' },
    tools: { label: 'Tools', icon: <Terminal />, color: '#764ba2' }
  };

  return (
    <motion.div 
      className="project-detail"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* Hero Section */}
      <div className="detail-hero" style={{
        background: `linear-gradient(135deg, ${project.color.primary}15, ${project.color.secondary}10)`
      }}>
        <div className="hero-backdrop">
          <div className="backdrop-pattern" />
          <motion.div 
            className="backdrop-glow"
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.5, 0.3]
            }}
            transition={{ duration: 5, repeat: Infinity }}
          />
        </div>

        <div className="hero-container">
          <motion.button 
            className="back-button"
            onClick={onBack}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <ArrowLeft size={20} />
            <span>Back to Projects</span>
          </motion.button>

          <motion.div 
            className="hero-content"
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <div className="hero-badges">
              {project.featured && (
                <span className="featured-badge">
                  <Star size={16} fill="currentColor" />
                  Featured
                </span>
              )}
              <span className="status-badge" style={{
                background: project.status === 'completed' ? '#00d4aa' : 
                          project.status === 'in-progress' ? '#ffd700' : '#667eea'
              }}>
                {project.status === 'completed' ? <CheckCircle size={16} /> :
                 project.status === 'in-progress' ? <Activity size={16} /> : <Clock size={16} />}
                {project.status.replace('-', ' ')}
              </span>
            </div>

            <h1 className="hero-title">{project.title}</h1>
            <p className="hero-subtitle">{project.subtitle}</p>
            
            <div className="hero-actions">
              {project.liveDemo && (
                <motion.a 
                  href={project.liveDemo}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="action-primary"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Play size={20} />
                  Live Demo
                </motion.a>
              )}
              {project.github && (
                <motion.a 
                  href={project.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="action-secondary"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Github size={20} />
                  View Source
                </motion.a>
              )}
              {project.documentation && (
                <motion.a 
                  href={project.documentation}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="action-secondary"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Download size={20} />
                  Documentation
                </motion.a>
              )}
            </div>

            <div className="hero-meta">
              <div className="meta-item">
                <Calendar size={16} />
                <span>Started {new Date(project.startDate).toLocaleDateString('en-US', { 
                  month: 'long', year: 'numeric' 
                })}</span>
              </div>
              {project.endDate && (
                <div className="meta-item">
                  <CheckCircle size={16} />
                  <span>Completed {new Date(project.endDate).toLocaleDateString('en-US', { 
                    month: 'long', year: 'numeric' 
                  })}</span>
                </div>
              )}
            </div>
          </motion.div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="metrics-section">
        <div className="metrics-container">
          <motion.h2 
            className="section-title"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            Key Metrics & Achievements
          </motion.h2>
          <div className="metrics-grid">
            {project.metrics.map((metric, index) => (
              <motion.div 
                key={metric.label}
                className="metric-card"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.4 + index * 0.1 }}
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: '0 10px 30px rgba(0,0,0,0.2)'
                }}
              >
                <div className="metric-icon">
                  <TrendingUp size={24} />
                </div>
                <div className="metric-content">
                  <div className="metric-value">{metric.value}</div>
                  <div className="metric-label">{metric.label}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="section-nav">
        <div className="nav-container">
          {sections.map(section => (
            <motion.button
              key={section.id}
              className={`nav-tab ${activeSection === section.id ? 'active' : ''}`}
              onClick={() => setActiveSection(section.id)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {section.icon}
              <span>{section.label}</span>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Content Sections */}
      <div className="content-sections">
        <AnimatePresence mode="wait">
          {/* Overview Section */}
          {activeSection === 'overview' && (
            <motion.div 
              key="overview"
              className="section-content"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <div className="overview-grid">
                <div className="overview-main">
                  <h3>Project Overview</h3>
                  <p className="overview-text">{project.longDescription}</p>
                  
                  {/* Image Gallery */}
                  {project.images && project.images.length > 0 && (
                    <div className="image-gallery">
                      <motion.div 
                        className="gallery-main"
                        onClick={() => setFullscreenImage(project.images[currentImageIndex])}
                        whileHover={{ scale: 1.02 }}
                      >
                        <img 
                          src={project.images[currentImageIndex]} 
                          alt={`${project.title} screenshot`}
                        />
                        <div className="gallery-overlay">
                          <Maximize2 size={24} />
                        </div>
                      </motion.div>
                      
                      {project.images.length > 1 && (
                        <div className="gallery-thumbs">
                          {project.images.map((img, index) => (
                            <motion.div
                              key={index}
                              className={`thumb ${currentImageIndex === index ? 'active' : ''}`}
                              onClick={() => setCurrentImageIndex(index)}
                              whileHover={{ scale: 1.1 }}
                              style={{ backgroundImage: `url(${img})` }}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className="overview-sidebar">
                  <div className="sidebar-card">
                    <h4>Quick Facts</h4>
                    <div className="facts-list">
                      <div className="fact">
                        <span className="fact-label">Category</span>
                        <span className="fact-value">{project.category.toUpperCase()}</span>
                      </div>
                      <div className="fact">
                        <span className="fact-label">Duration</span>
                        <span className="fact-value">
                          {project.endDate ? 
                            Math.ceil((new Date(project.endDate).getTime() - new Date(project.startDate).getTime()) / (1000 * 60 * 60 * 24 * 30)) + ' months' :
                            'Ongoing'
                          }
                        </span>
                      </div>
                      <div className="fact">
                        <span className="fact-label">Technologies</span>
                        <span className="fact-value">{project.techStack.length} tools</span>
                      </div>
                    </div>
                  </div>

                  <div className="sidebar-card">
                    <h4>Tags</h4>
                    <div className="tags-cloud">
                      {project.tags.map(tag => (
                        <span key={tag} className="tag-item">{tag}</span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Tech Stack Section */}
          {activeSection === 'tech' && (
            <motion.div 
              key="tech"
              className="section-content"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h3>Technology Stack</h3>
              
              <div className="tech-categories">
                {Object.entries(techCategories).map(([category, config]) => {
                  const techs = project.techStack.filter(t => t.category === category);
                  if (techs.length === 0) return null;
                  
                  return (
                    <motion.div 
                      key={category}
                      className="tech-category"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                    >
                      <div className="category-header" style={{ color: config.color }}>
                        {config.icon}
                        <h4>{config.label}</h4>
                      </div>
                      
                      <div className="tech-list">
                        {techs.map((tech, index) => (
                          <motion.div 
                            key={tech.name}
                            className="tech-item"
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.1 }}
                          >
                            <div className="tech-info">
                              <span className="tech-name">{tech.name}</span>
                              <span className="tech-proficiency">{tech.proficiency}%</span>
                            </div>
                            <div className="proficiency-bar">
                              <motion.div 
                                className="proficiency-fill"
                                initial={{ width: 0 }}
                                animate={{ width: `${tech.proficiency}%` }}
                                transition={{ delay: 0.5 + index * 0.1, duration: 0.8 }}
                                style={{ background: config.color }}
                              />
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {/* Features Section */}
          {activeSection === 'features' && (
            <motion.div 
              key="features"
              className="section-content"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h3>Key Features</h3>
              
              <div className="features-grid">
                {project.features.map((feature, index) => (
                  <motion.div 
                    key={feature.title}
                    className="feature-card"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.03, boxShadow: '0 10px 30px rgba(0,0,0,0.2)' }}
                  >
                    <div className="feature-icon">
                      <Zap size={24} />
                    </div>
                    <h4>{feature.title}</h4>
                    <p>{feature.description}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Challenges Section */}
          {activeSection === 'challenges' && (
            <motion.div 
              key="challenges"
              className="section-content"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h3>Challenges & Solutions</h3>
              
              <div className="challenges-solutions">
                {project.challenges.map((challenge, index) => (
                  <motion.div 
                    key={index}
                    className="challenge-item"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.15 }}
                  >
                    <div className="challenge">
                      <div className="challenge-header">
                        <Shield size={20} style={{ color: '#ff6b6b' }} />
                        <h4>Challenge #{index + 1}</h4>
                      </div>
                      <p>{challenge}</p>
                    </div>
                    
                    <div className="solution-arrow">
                      <ChevronRight size={24} />
                    </div>
                    
                    <div className="solution">
                      <div className="solution-header">
                        <CheckCircle size={20} style={{ color: '#00d4aa' }} />
                        <h4>Solution</h4>
                      </div>
                      <p>{project.solutions[index]}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Impact Section */}
          {activeSection === 'impact' && (
            <motion.div 
              key="impact"
              className="section-content"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h3>Project Impact</h3>
              
              <div className="impact-grid">
                {project.impact.map((impact, index) => (
                  <motion.div 
                    key={index}
                    className="impact-card"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.05 }}
                  >
                    <div className="impact-number">{index + 1}</div>
                    <p>{impact}</p>
                  </motion.div>
                ))}
              </div>

              {project.testimonials && (
                <div className="testimonials">
                  <h4>What Others Say</h4>
                  <div className="testimonials-grid">
                    {project.testimonials.map((testimonial, index) => (
                      <motion.div 
                        key={index}
                        className="testimonial"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.1 }}
                      >
                        <p className="testimonial-content">"{testimonial.content}"</p>
                        <div className="testimonial-author">
                          <div className="author-avatar">
                            {testimonial.avatar ? 
                              <img src={testimonial.avatar} alt={testimonial.name} /> :
                              <div className="avatar-placeholder">{testimonial.name[0]}</div>
                            }
                          </div>
                          <div className="author-info">
                            <div className="author-name">{testimonial.name}</div>
                            <div className="author-role">{testimonial.role}</div>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Fullscreen Image Viewer */}
      <AnimatePresence>
        {fullscreenImage && (
          <motion.div 
            className="fullscreen-viewer"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setFullscreenImage(null)}
          >
            <button 
              className="close-viewer"
              onClick={() => setFullscreenImage(null)}
            >
              <X size={24} />
            </button>
            <img src={fullscreenImage} alt="Fullscreen view" />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default ProjectDetail;