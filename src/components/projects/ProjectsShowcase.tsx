// src/components/projects/ProjectsShowcase.tsx
import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Code2, Brain, Database, Globe, Smartphone,
  Filter, Search, Star, Github, ExternalLink,
  Clock, CheckCircle, Activity, TrendingUp,
  Zap, Users, Award, Target, BarChart3,
  ChevronRight, Sparkles, Play, Eye
} from 'lucide-react';
import { projectsData, type Project } from '../../data/projectsData';
import './ProjectsShowcase.css';

interface ProjectsShowcaseProps {
  onProjectSelect?: (projectId: string) => void;
}

const ProjectsShowcase: React.FC<ProjectsShowcaseProps> = ({ onProjectSelect }) => {
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [hoveredProject, setHoveredProject] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'status'>('date');
  const [showFilters, setShowFilters] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const categories = [
    { id: 'all', label: 'All Projects', icon: <Code2 />, color: '#667eea' },
    { id: 'ml', label: 'Machine Learning', icon: <Brain />, color: '#f093fb' },
    { id: 'web', label: 'Web Development', icon: <Globe />, color: '#00d4aa' },
    { id: 'data', label: 'Data Science', icon: <Database />, color: '#ffd700' },
    { id: 'mobile', label: 'Mobile Apps', icon: <Smartphone />, color: '#ff6b6b' },
  ];

  const statusColors = {
    'in-progress': { bg: '#ffd700', text: 'In Progress', icon: <Activity /> },
    'completed': { bg: '#00d4aa', text: 'Completed', icon: <CheckCircle /> },
    'maintained': { bg: '#667eea', text: 'Maintained', icon: <Clock /> }
  };

  const filteredProjects = projectsData
    .filter(project => {
      const matchesCategory = selectedCategory === 'all' || project.category === selectedCategory;
      const matchesSearch = project.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           project.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           project.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesCategory && matchesSearch;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.title.localeCompare(b.title);
        case 'status':
          return a.status.localeCompare(b.status);
        case 'date':
        default:
          return new Date(b.startDate).getTime() - new Date(a.startDate).getTime();
      }
    });

  const featuredProject = filteredProjects.find(p => p.featured);
  const regularProjects = filteredProjects.filter(p => !p.featured);

  useEffect(() => {
    // Parallax effect for featured project
    const handleScroll = () => {
      if (containerRef.current) {
        const scrolled = window.scrollY;
        const parallaxElements = containerRef.current.querySelectorAll('.parallax-bg');
        parallaxElements.forEach((el: any) => {
          el.style.transform = `translateY(${scrolled * 0.5}px)`;
        });
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const ProjectCard: React.FC<{ project: Project; index: number }> = ({ project, index }) => {
    const isHovered = hoveredProject === project.id;
    
    return (
      <motion.div
        className={`project-card ${isHovered ? 'hovered' : ''}`}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1, duration: 0.5 }}
        onMouseEnter={() => setHoveredProject(project.id)}
        onMouseLeave={() => setHoveredProject(null)}
        onClick={() => onProjectSelect?.(project.id)}
        layout
      >
        {/* Background gradient effect */}
        <div 
          className="card-gradient"
          style={{
            background: `linear-gradient(135deg, ${project.color.primary}20, ${project.color.secondary}10)`,
            opacity: isHovered ? 1 : 0
          }}
        />

        {/* Status Badge */}
        <div className="status-badge" style={{ background: statusColors[project.status].bg }}>
          {statusColors[project.status].icon}
          <span>{statusColors[project.status].text}</span>
        </div>

        {/* Thumbnail with overlay */}
        <div className="project-thumbnail">
          <div className="thumbnail-overlay">
            <motion.div
              className="overlay-content"
              initial={{ opacity: 0 }}
              animate={{ opacity: isHovered ? 1 : 0 }}
              transition={{ duration: 0.3 }}
            >
              <button className="view-btn">
                <Eye size={20} />
                <span>View Details</span>
              </button>
              {project.liveDemo && (
                <a href={project.liveDemo} target="_blank" rel="noopener noreferrer" 
                   onClick={(e) => e.stopPropagation()} className="demo-btn">
                  <Play size={20} />
                  <span>Live Demo</span>
                </a>
              )}
            </motion.div>
          </div>
          
          {/* Tech stack floating badges */}
          <div className="tech-badges">
            {project.techStack.slice(0, 3).map((tech, i) => (
              <motion.div
                key={tech.name}
                className="tech-badge"
                initial={{ scale: 0 }}
                animate={{ scale: isHovered ? 1 : 0 }}
                transition={{ delay: i * 0.1 }}
              >
                {tech.name}
              </motion.div>
            ))}
            {project.techStack.length > 3 && (
              <motion.div
                className="tech-badge more"
                initial={{ scale: 0 }}
                animate={{ scale: isHovered ? 1 : 0 }}
                transition={{ delay: 0.3 }}
              >
                +{project.techStack.length - 3}
              </motion.div>
            )}
          </div>
        </div>

        {/* Content */}
        <div className="project-content">
          <h3 className="project-title">
            {project.title}
            {project.featured && (
              <span className="featured-star">
                <Star size={16} fill="currentColor" />
              </span>
            )}
          </h3>
          <p className="project-subtitle">{project.subtitle}</p>
          <p className="project-description">{project.description}</p>

          {/* Metrics */}
          <div className="project-metrics">
            {project.metrics.slice(0, 3).map((metric) => (
              <div key={metric.label} className="metric">
                <div className="metric-value">{metric.value}</div>
                <div className="metric-label">{metric.label}</div>
              </div>
            ))}
          </div>

          {/* Tags */}
          <div className="project-tags">
            {project.tags.slice(0, 4).map(tag => (
              <span key={tag} className="tag">{tag}</span>
            ))}
          </div>

          {/* Actions */}
          <div className="project-actions">
            {project.github && (
              <a href={project.github} target="_blank" rel="noopener noreferrer"
                 onClick={(e) => e.stopPropagation()} className="action-link">
                <Github size={18} />
              </a>
            )}
            {project.liveDemo && (
              <a href={project.liveDemo} target="_blank" rel="noopener noreferrer"
                 onClick={(e) => e.stopPropagation()} className="action-link">
                <ExternalLink size={18} />
              </a>
            )}
            <button className="action-link" onClick={() => onProjectSelect?.(project.id)}>
              <ChevronRight size={18} />
            </button>
          </div>
        </div>

        {/* Hover effect particles */}
        {isHovered && (
          <div className="hover-particles">
            {[...Array(5)].map((_, i) => (
              <motion.div
                key={i}
                className="particle"
                initial={{ scale: 0, x: 0, y: 0 }}
                animate={{
                  scale: [0, 1, 0],
                  x: Math.random() * 100 - 50,
                  y: Math.random() * 100 - 50,
                }}
                transition={{
                  duration: 1,
                  delay: i * 0.1,
                  repeat: Infinity,
                }}
              />
            ))}
          </div>
        )}
      </motion.div>
    );
  };

  return (
    <div className="projects-showcase" ref={containerRef}>
      {/* Hero Section */}
      <motion.div 
        className="showcase-hero"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
      >
        <div className="hero-background">
          <div className="hero-pattern" />
          <div className="parallax-bg" />
        </div>
        
        <div className="hero-content">
          <motion.h1
            className="hero-title"
            initial={{ y: 30, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Sparkles className="title-icon" />
            Projects that Push Boundaries
          </motion.h1>
          <motion.p
            className="hero-subtitle"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            Explore my portfolio of innovative solutions, from ML-powered platforms to stunning web experiences
          </motion.p>

          {/* Stats */}
          <motion.div
            className="hero-stats"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <div className="stat">
              <div className="stat-number">{projectsData.length}</div>
              <div className="stat-label">Projects</div>
            </div>
            <div className="stat">
              <div className="stat-number">20+</div>
              <div className="stat-label">Technologies</div>
            </div>
            <div className="stat">
              <div className="stat-number">100K+</div>
              <div className="stat-label">Lines of Code</div>
            </div>
            <div className="stat">
              <div className="stat-number">∞</div>
              <div className="stat-label">Passion</div>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Filters and Search */}
      <div className="showcase-controls">
        <div className="controls-wrapper">
          {/* Search */}
          <div className="search-box">
            <Search size={20} />
            <input
              type="text"
              placeholder="Search projects, technologies..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Categories */}
          <div className="category-filters">
            {categories.map(cat => (
              <motion.button
                key={cat.id}
                className={`category-btn ${selectedCategory === cat.id ? 'active' : ''}`}
                onClick={() => setSelectedCategory(cat.id)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                style={{
                  '--cat-color': cat.color
                } as React.CSSProperties}
              >
                {cat.icon}
                <span>{cat.label}</span>
              </motion.button>
            ))}
          </div>

          {/* View Options */}
          <div className="view-options">
            <button
              className="filter-toggle"
              onClick={() => setShowFilters(!showFilters)}
            >
              <Filter size={18} />
              <span>Filters</span>
            </button>
            
            <div className="view-mode">
              <button
                className={viewMode === 'grid' ? 'active' : ''}
                onClick={() => setViewMode('grid')}
              >
                <BarChart3 size={18} />
              </button>
              <button
                className={viewMode === 'list' ? 'active' : ''}
                onClick={() => setViewMode('list')}
              >
                <TrendingUp size={18} />
              </button>
            </div>
          </div>
        </div>

        {/* Advanced Filters */}
        <AnimatePresence>
          {showFilters && (
            <motion.div
              className="advanced-filters"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
            >
              <div className="filter-group">
                <label>Sort By</label>
                <select value={sortBy} onChange={(e) => setSortBy(e.target.value as any)}>
                  <option value="date">Latest First</option>
                  <option value="name">Alphabetical</option>
                  <option value="status">Status</option>
                </select>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Featured Project */}
      {featuredProject && (
        <motion.div 
          className="featured-project"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        >
          <div className="featured-badge">
            <Star size={20} fill="currentColor" />
            <span>Featured Project</span>
          </div>
          
          <div className="featured-content">
            <div className="featured-info">
              <h2 className="featured-title">{featuredProject.title}</h2>
              <p className="featured-subtitle">{featuredProject.subtitle}</p>
              <p className="featured-description">{featuredProject.longDescription}</p>
              
              <div className="featured-metrics">
                {featuredProject.metrics.map(metric => (
                  <div key={metric.label} className="featured-metric">
                    <Zap className="metric-icon" />
                    <div>
                      <div className="metric-value">{metric.value}</div>
                      <div className="metric-label">{metric.label}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="featured-actions">
                <button 
                  className="primary-btn"
                  onClick={() => onProjectSelect?.(featuredProject.id)}
                >
                  <Eye size={20} />
                  View Case Study
                </button>
                {featuredProject.liveDemo && (
                  <a 
                    href={featuredProject.liveDemo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="secondary-btn"
                  >
                    <Play size={20} />
                    Live Demo
                  </a>
                )}
              </div>
            </div>
            
            <div className="featured-visual">
              <div className="visual-container">
                <div className="code-preview">
                  {featuredProject.codeSnippets?.[0] && (
                    <pre>
                      <code>{featuredProject.codeSnippets[0].code}</code>
                    </pre>
                  )}
                </div>
                <div className="tech-stack-visual">
                  {featuredProject.techStack.map((tech, i) => (
                    <motion.div
                      key={tech.name}
                      className="tech-item"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                    >
                      <div className="tech-name">{tech.name}</div>
                      <div className="tech-bar">
                        <motion.div
                          className="tech-fill"
                          initial={{ width: 0 }}
                          animate={{ width: `${tech.proficiency}%` }}
                          transition={{ delay: 0.5 + i * 0.1, duration: 0.8 }}
                          style={{ background: featuredProject.color.primary }}
                        />
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Projects Grid */}
      <div className={`projects-grid ${viewMode}`}>
        <AnimatePresence mode="popLayout">
          {regularProjects.map((project, index) => (
            <ProjectCard key={project.id} project={project} index={index} />
          ))}
        </AnimatePresence>
      </div>

      {/* Empty State */}
      {filteredProjects.length === 0 && (
        <motion.div 
          className="empty-state"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <Search size={48} />
          <h3>No projects found</h3>
          <p>Try adjusting your filters or search query</p>
        </motion.div>
      )}
    </div>
  );
};

export default ProjectsShowcase;