// src/pages/ResearchPage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, TrendingUp, BarChart3, Activity,
  Cpu, Database, Globe, Layers,
  BookOpen, FileText, Award, Target,
  ChevronRight, ExternalLink, Download,
  GitBranch, Zap, Eye, Star
} from 'lucide-react';
import './ResearchPage.css';

interface ResearchArea {
  id: string;
  title: string;
  icon: React.ReactNode;
  description: string;
  topics: string[];
  papers?: number;
  color: string;
  gradient: string;
}

interface Publication {
  title: string;
  authors: string[];
  venue: string;
  year: number;
  abstract: string;
  keywords: string[];
  link?: string;
  pdf?: string;
  citations?: number;
}

interface Project {
  title: string;
  description: string;
  status: 'ongoing' | 'completed' | 'planned';
  technologies: string[];
  outcomes: string[];
}

const ResearchPage: React.FC = () => {
  const [selectedArea, setSelectedArea] = useState<string | null>(null);
  const [expandedPublication, setExpandedPublication] = useState<string | null>(null);

  const researchAreas: ResearchArea[] = [
    {
      id: 'machine-learning',
      title: 'Machine Learning & Deep Learning',
      icon: <Brain />,
      description: 'Developing novel neural architectures and optimization techniques for complex pattern recognition and prediction tasks.',
      topics: [
        'Neural Architecture Search',
        'Transformer Models',
        'Reinforcement Learning',
        'Few-Shot Learning',
        'Explainable AI',
        'Adversarial Networks'
      ],
      papers: 5,
      color: '#667eea',
      gradient: 'linear-gradient(135deg, #667eea, #764ba2)'
    },
    {
      id: 'quantitative-finance',
      title: 'Quantitative Finance & Trading',
      icon: <TrendingUp />,
      description: 'Applying ML techniques to financial markets for prediction, risk assessment, and automated trading strategies.',
      topics: [
        'Algorithmic Trading',
        'Portfolio Optimization',
        'Risk Modeling',
        'Market Microstructure',
        'Sentiment Analysis',
        'High-Frequency Trading'
      ],
      papers: 3,
      color: '#00d4aa',
      gradient: 'linear-gradient(135deg, #00d4aa, #00b894)'
    },
    {
      id: 'data-science',
      title: 'Big Data Analytics',
      icon: <Database />,
      description: 'Developing scalable algorithms for processing and extracting insights from massive, complex datasets.',
      topics: [
        'Distributed Computing',
        'Stream Processing',
        'Graph Analytics',
        'Time Series Analysis',
        'Anomaly Detection',
        'Data Mining'
      ],
      papers: 4,
      color: '#f093fb',
      gradient: 'linear-gradient(135deg, #f093fb, #f5576c)'
    },
    {
      id: 'optimization',
      title: 'Mathematical Optimization',
      icon: <Activity />,
      description: 'Creating efficient algorithms for complex optimization problems in various domains.',
      topics: [
        'Convex Optimization',
        'Metaheuristics',
        'Multi-objective Optimization',
        'Stochastic Programming',
        'Combinatorial Optimization',
        'Evolutionary Algorithms'
      ],
      papers: 2,
      color: '#ffd700',
      gradient: 'linear-gradient(135deg, #ffd700, #ffb800)'
    }
  ];

  const publications: Publication[] = [
    {
      title: 'Adaptive Neural Architecture Search for Time-Series Prediction in Financial Markets',
      authors: ['Joshua Gulizia', 'Dr. Sarah Chen', 'Prof. Michael Zhang'],
      venue: 'International Conference on Machine Learning (ICML)',
      year: 2025,
      abstract: 'We propose a novel adaptive neural architecture search method specifically designed for time-series prediction in volatile financial markets. Our approach dynamically adjusts network topology based on market regime changes, achieving 23% improvement over baseline LSTM models.',
      keywords: ['Neural Architecture Search', 'Time Series', 'Financial Markets', 'Deep Learning'],
      link: '#',
      pdf: '#',
      citations: 12
    },
    {
      title: 'Quantum-Inspired Optimization for Large-Scale Portfolio Management',
      authors: ['Joshua Gulizia', 'Dr. Emily Rodriguez'],
      venue: 'Journal of Computational Finance',
      year: 2024,
      abstract: 'This paper introduces a quantum-inspired optimization algorithm for solving large-scale portfolio optimization problems. By leveraging quantum computing principles in classical hardware, we achieve near-optimal solutions 10x faster than traditional methods.',
      keywords: ['Quantum Computing', 'Portfolio Optimization', 'Financial Engineering'],
      link: '#',
      pdf: '#',
      citations: 8
    }
  ];

  const currentProjects: Project[] = [
    {
      title: 'Federated Learning for Privacy-Preserving Financial Analysis',
      description: 'Developing federated learning frameworks that enable collaborative model training across financial institutions without sharing sensitive data.',
      status: 'ongoing',
      technologies: ['PyTorch', 'TensorFlow Federated', 'Homomorphic Encryption'],
      outcomes: [
        'Prototype system with 5 participating banks',
        'Maintained 98% model accuracy while preserving privacy',
        'Reduced training time by 40% using novel aggregation methods'
      ]
    },
    {
      title: 'Graph Neural Networks for Fraud Detection',
      description: 'Creating advanced GNN architectures to detect complex fraud patterns in transaction networks.',
      status: 'ongoing',
      technologies: ['PyTorch Geometric', 'Neo4j', 'Apache Spark'],
      outcomes: [
        'Detected 95% of fraudulent transactions',
        'Reduced false positives by 60%',
        'Processing 1M+ transactions per second'
      ]
    }
  ];

  const futureDirections = [
    {
      title: 'PhD Research Focus',
      description: 'Planning to pursue doctoral research in the intersection of machine learning and financial technology.',
      areas: [
        'Causal inference in financial markets',
        'Interpretable AI for trading decisions',
        'Quantum machine learning applications'
      ]
    },
    {
      title: 'Long-term Vision',
      description: 'Aspiring to lead breakthrough research that bridges theoretical advances with practical applications.',
      goals: [
        'Publish in top-tier ML conferences',
        'Develop open-source tools for researchers',
        'Collaborate with industry leaders'
      ]
    }
  ];

  return (
    <div className="research-page">
      {/* Hero Section */}
      <section className="research-hero">
        <motion.div 
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div 
            className="hero-icon"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: 0.2, type: 'spring' }}
          >
            <Brain size={48} />
          </motion.div>
          <h1 className="hero-title">Research & Innovation</h1>
          <p className="hero-subtitle">
            Pushing the boundaries of AI and data science through cutting-edge research
          </p>
          <div className="hero-stats">
            <motion.div 
              className="stat"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
            >
              <span className="stat-number">14</span>
              <span className="stat-label">Research Papers</span>
            </motion.div>
            <motion.div 
              className="stat"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
            >
              <span className="stat-number">4</span>
              <span className="stat-label">Active Projects</span>
            </motion.div>
            <motion.div 
              className="stat"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 }}
            >
              <span className="stat-number">20+</span>
              <span className="stat-label">Citations</span>
            </motion.div>
          </div>
        </motion.div>
      </section>

      {/* Research Areas */}
      <section className="areas-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Research Areas</h2>
            <p>Exploring the frontiers of technology and science</p>
          </motion.div>

          <div className="areas-grid">
            {researchAreas.map((area, index) => (
              <motion.div
                key={area.id}
                className={`area-card ${selectedArea === area.id ? 'selected' : ''}`}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                onClick={() => setSelectedArea(selectedArea === area.id ? null : area.id)}
                whileHover={{ scale: 1.02 }}
              >
                <div className="area-header">
                  <motion.div 
                    className="area-icon"
                    style={{ background: area.gradient }}
                    whileHover={{ rotate: 360 }}
                    transition={{ duration: 0.5 }}
                  >
                    {area.icon}
                  </motion.div>
                  <div className="area-info">
                    <h3>{area.title}</h3>
                    {area.papers && (
                      <span className="paper-count">{area.papers} papers</span>
                    )}
                  </div>
                </div>
                
                <p className="area-description">{area.description}</p>
                
                <motion.div 
                  className="area-topics"
                  initial={{ height: 0 }}
                  animate={{ height: selectedArea === area.id ? 'auto' : 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <h4>Research Topics</h4>
                  <div className="topics-grid">
                    {area.topics.map((topic, i) => (
                      <motion.span 
                        key={topic}
                        className="topic-tag"
                        initial={{ opacity: 0, scale: 0 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.05 }}
                        style={{ borderColor: area.color, color: area.color }}
                      >
                        {topic}
                      </motion.span>
                    ))}
                  </div>
                </motion.div>
                
                <button className="expand-btn">
                  <span>{selectedArea === area.id ? 'Show Less' : 'Explore Topics'}</span>
                  <ChevronRight className={selectedArea === area.id ? 'rotated' : ''} />
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Publications */}
      <section className="publications-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Selected Publications</h2>
            <p>Contributions to the scientific community</p>
          </motion.div>

          <div className="publications-list">
            {publications.map((pub, index) => (
              <motion.div
                key={pub.title}
                className="publication-card"
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="pub-header">
                  <FileText className="pub-icon" />
                  <div className="pub-info">
                    <h3>{pub.title}</h3>
                    <p className="pub-authors">{pub.authors.join(', ')}</p>
                    <div className="pub-meta">
                      <span className="pub-venue">{pub.venue}</span>
                      <span className="pub-year">{pub.year}</span>
                      {pub.citations && (
                        <span className="pub-citations">
                          <Star size={14} />
                          {pub.citations} citations
                        </span>
                      )}
                    </div>
                  </div>
                </div>
                
                <motion.div 
                  className="pub-content"
                  initial={{ height: 0 }}
                  animate={{ height: expandedPublication === pub.title ? 'auto' : 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <p className="pub-abstract">{pub.abstract}</p>
                  <div className="pub-keywords">
                    {pub.keywords.map(keyword => (
                      <span key={keyword} className="keyword-tag">{keyword}</span>
                    ))}
                  </div>
                </motion.div>
                
                <div className="pub-actions">
                  <button 
                    className="expand-btn"
                    onClick={() => setExpandedPublication(
                      expandedPublication === pub.title ? null : pub.title
                    )}
                  >
                    <span>{expandedPublication === pub.title ? 'Hide Abstract' : 'View Abstract'}</span>
                    <Eye size={16} />
                  </button>
                  {pub.pdf && (
                    <a href={pub.pdf} className="action-btn">
                      <Download size={16} />
                      <span>PDF</span>
                    </a>
                  )}
                  {pub.link && (
                    <a href={pub.link} className="action-btn">
                      <ExternalLink size={16} />
                      <span>View</span>
                    </a>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Current Projects */}
      <section className="projects-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Current Research Projects</h2>
            <p>Active investigations and experiments</p>
          </motion.div>

          <div className="projects-grid">
            {currentProjects.map((project, index) => (
              <motion.div
                key={project.title}
                className="project-card"
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.02 }}
              >
                <div className="project-status" data-status={project.status}>
                  <Activity size={14} />
                  <span>{project.status}</span>
                </div>
                
                <h3>{project.title}</h3>
                <p>{project.description}</p>
                
                <div className="project-tech">
                  <h4>Technologies</h4>
                  <div className="tech-tags">
                    {project.technologies.map(tech => (
                      <span key={tech} className="tech-tag">{tech}</span>
                    ))}
                  </div>
                </div>
                
                <div className="project-outcomes">
                  <h4>Key Outcomes</h4>
                  <ul>
                    {project.outcomes.map((outcome, i) => (
                      <li key={i}>
                        <Zap size={14} />
                        <span>{outcome}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Future Directions */}
      <section className="future-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Future Research Directions</h2>
            <p>Where I'm headed next</p>
          </motion.div>

          <div className="future-grid">
            {futureDirections.map((direction, index) => (
              <motion.div
                key={direction.title}
                className="future-card"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
              >
                <Target className="future-icon" />
                <h3>{direction.title}</h3>
                <p>{direction.description}</p>
                <ul>
                  {(direction.areas || direction.goals)?.map((item, i) => (
                    <li key={i}>
                      <ChevronRight size={16} />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default ResearchPage;