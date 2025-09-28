// src/pages/ExperiencePage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Briefcase, Calendar, MapPin, ChevronRight,
  Database, Code2, TrendingUp, Users, 
  CheckCircle, Award, Zap, Target,
  BookOpen, GitBranch, Terminal, Globe
} from 'lucide-react';
import './ExperiencePage.css';

interface Experience {
  id: string;
  title: string;
  company: string;
  location: string;
  startDate: string;
  endDate: string;
  current: boolean;
  type: 'full-time' | 'part-time' | 'internship' | 'leadership';
  description: string;
  achievements: string[];
  technologies: string[];
  icon: React.ReactNode;
  color: string;
}

const ExperiencePage: React.FC = () => {
  const [selectedExperience, setSelectedExperience] = useState<string | null>(null);

  const experiences: Experience[] = [
    {
      id: 'data-science-intern',
      title: 'Data Science Intern',
      company: 'Sun & Ski Sports',
      location: 'Houston, TX',
      startDate: 'Aug 2025',
      endDate: 'Present',
      current: true,
      type: 'internship',
      description: 'Leading data analytics initiatives for a major sports retail chain with 40+ locations.',
      achievements: [
        'Analyzed 10M+ rows of sales data using PostgreSQL and pandas across 40 retail locations',
        'Created 5 Power BI dashboards for performance metrics, reducing weekly reporting time by 80%',
        'Streamlined data cleaning using Python, resolving 1,700+ data inconsistencies and anomalies',
        'Implemented automated ETL pipelines processing 100K+ daily transactions',
        'Developed predictive models for inventory optimization with 85% accuracy'
      ],
      technologies: ['Python', 'PostgreSQL', 'Power BI', 'Pandas', 'NumPy', 'Scikit-learn', 'ETL', 'Azure'],
      icon: <Database />,
      color: '#667eea'
    },
    {
      id: 'operations-officer',
      title: 'Operations Officer',
      company: 'Code Coogs',
      location: 'University of Houston',
      startDate: 'Jan 2025',
      endDate: 'Present',
      current: true,
      type: 'leadership',
      description: 'Leading technical education initiatives for the university\'s premier coding organization.',
      achievements: [
        'Conduct technical workshops teaching full-stack development to 200+ students per semester',
        'Collaborate with officers to coordinate workshop scheduling and curriculum development',
        'Mentor students on programming fundamentals, project development, and industry best practices',
        'Organized hackathon events with 100+ participants and industry sponsors',
        'Developed comprehensive workshop materials covering React, Node.js, and cloud deployment'
      ],
      technologies: ['React', 'Node.js', 'JavaScript', 'Python', 'Git', 'AWS', 'Docker'],
      icon: <Users />,
      color: '#f093fb'
    },
    {
      id: 'bookkeeper',
      title: 'QuickBooks Bookkeeper',
      company: 'Reel It Inn Rentals',
      location: 'Freeport, TX',
      startDate: 'Apr 2023',
      endDate: 'Jul 2025',
      current: false,
      type: 'part-time',
      description: 'Managed complete financial operations for a vacation rental business.',
      achievements: [
        'Managed $60K yearly revenue processing 50+ monthly transactions in QuickBooks',
        'Maintained financial records with 99% accuracy for accounts payable and receivable',
        'Generated monthly financial reports to track revenue trends and inform management decisions',
        'Automated invoice generation and payment tracking, saving 10 hours weekly',
        'Implemented digital filing system for improved document management'
      ],
      technologies: ['QuickBooks', 'Excel', 'Financial Reporting', 'Data Analysis', 'Automation'],
      icon: <TrendingUp />,
      color: '#00d4aa'
    }
  ];

  const skills = {
    technical: [
      { name: 'Python', level: 95 },
      { name: 'React', level: 90 },
      { name: 'SQL', level: 88 },
      { name: 'Machine Learning', level: 85 },
      { name: 'Data Visualization', level: 92 },
      { name: 'Cloud Computing', level: 80 }
    ],
    soft: [
      'Problem Solving',
      'Team Leadership',
      'Communication',
      'Project Management',
      'Mentoring',
      'Critical Thinking'
    ]
  };

  const achievements = [
    {
      icon: <Award />,
      title: 'Dean\'s List',
      description: 'Maintained 3.8+ GPA for multiple semesters',
      date: '2023-2025'
    },
    {
      icon: <Trophy />,
      title: 'Hackathon Winner',
      description: 'First place in UH Hackathon 2024',
      date: '2024'
    },
    {
      icon: <Star />,
      title: 'Top Performer',
      description: 'Recognized for exceptional data analysis work',
      date: '2025'
    }
  ];

  const Trophy = () => (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6m12 0h1.5a2.5 2.5 0 0 1 0 5H18M6 4h12v11a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V4Z"/>
      <path d="M12 19v3m-4 0h8"/>
    </svg>
  );

  const Star = () => (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor">
      <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
    </svg>
  );

  return (
    <div className="experience-page">
      {/* Hero Section */}
      <section className="experience-hero">
        <motion.div 
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div 
            className="hero-icon"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring' }}
          >
            <Briefcase size={48} />
          </motion.div>
          <h1 className="hero-title">Professional Experience</h1>
          <p className="hero-subtitle">
            Building impactful solutions through data science, leadership, and innovation
          </p>
        </motion.div>
      </section>

      {/* Timeline Section */}
      <section className="timeline-section">
        <div className="container">
          <motion.div 
            className="timeline-container"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
          >
            {experiences.map((exp, index) => (
              <motion.div 
                key={exp.id}
                className={`timeline-item ${exp.current ? 'current' : ''} ${selectedExperience === exp.id ? 'selected' : ''}`}
                initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
                onClick={() => setSelectedExperience(selectedExperience === exp.id ? null : exp.id)}
              >
                <motion.div 
                  className="timeline-dot"
                  style={{ background: exp.color }}
                  whileHover={{ scale: 1.2 }}
                >
                  {exp.icon}
                </motion.div>
                
                <motion.div 
                  className="timeline-content"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="content-header">
                    <div className="header-info">
                      <h3>{exp.title}</h3>
                      <p className="company">{exp.company}</p>
                      <div className="meta">
                        <span className="location">
                          <MapPin size={14} />
                          {exp.location}
                        </span>
                        <span className="date">
                          <Calendar size={14} />
                          {exp.startDate} - {exp.endDate}
                        </span>
                      </div>
                    </div>
                    {exp.current && (
                      <span className="current-badge">Current</span>
                    )}
                  </div>
                  
                  <p className="description">{exp.description}</p>
                  
                  <motion.div 
                    className="expanded-content"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ 
                      height: selectedExperience === exp.id ? 'auto' : 0,
                      opacity: selectedExperience === exp.id ? 1 : 0
                    }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="achievements">
                      <h4>Key Achievements</h4>
                      <ul>
                        {exp.achievements.map((achievement, i) => (
                          <motion.li 
                            key={i}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.1 }}
                          >
                            <CheckCircle size={16} />
                            <span>{achievement}</span>
                          </motion.li>
                        ))}
                      </ul>
                    </div>
                    
                    <div className="tech-stack">
                      <h4>Technologies Used</h4>
                      <div className="tech-tags">
                        {exp.technologies.map((tech, i) => (
                          <motion.span 
                            key={tech}
                            className="tech-tag"
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: i * 0.05 }}
                            whileHover={{ scale: 1.1 }}
                          >
                            {tech}
                          </motion.span>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                  
                  <button className="expand-btn">
                    <span>{selectedExperience === exp.id ? 'Show Less' : 'Show More'}</span>
                    <ChevronRight className={selectedExperience === exp.id ? 'rotated' : ''} />
                  </button>
                </motion.div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="skills-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Technical Expertise</h2>
            <p>Technologies and skills I've mastered throughout my journey</p>
          </motion.div>
          
          <div className="skills-grid">
            <motion.div 
              className="skills-column"
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h3>
                <Terminal size={24} />
                Technical Skills
              </h3>
              <div className="skill-bars">
                {skills.technical.map((skill, index) => (
                  <motion.div 
                    key={skill.name}
                    className="skill-item"
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <div className="skill-header">
                      <span className="skill-name">{skill.name}</span>
                      <span className="skill-level">{skill.level}%</span>
                    </div>
                    <div className="skill-bar">
                      <motion.div 
                        className="skill-progress"
                        initial={{ width: 0 }}
                        whileInView={{ width: `${skill.level}%` }}
                        viewport={{ once: true }}
                        transition={{ duration: 1, delay: 0.5 + index * 0.1 }}
                      />
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
            
            <motion.div 
              className="skills-column"
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
            >
              <h3>
                <Users size={24} />
                Soft Skills
              </h3>
              <div className="soft-skills">
                {skills.soft.map((skill, index) => (
                  <motion.span 
                    key={skill}
                    className="soft-skill-tag"
                    initial={{ opacity: 0, scale: 0 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.1 }}
                  >
                    {skill}
                  </motion.span>
                ))}
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Achievements Section */}
      <section className="achievements-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Achievements & Recognition</h2>
            <p>Milestones and accomplishments throughout my career</p>
          </motion.div>
          
          <div className="achievements-grid">
            {achievements.map((achievement, index) => (
              <motion.div 
                key={achievement.title}
                className="achievement-card"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05, rotateY: 10 }}
              >
                <motion.div 
                  className="achievement-icon"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                >
                  {achievement.icon}
                </motion.div>
                <h3>{achievement.title}</h3>
                <p>{achievement.description}</p>
                <span className="achievement-date">{achievement.date}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default ExperiencePage;