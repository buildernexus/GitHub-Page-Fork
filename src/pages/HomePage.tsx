// src/pages/HomePage.tsx
import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useSpring, useInView } from 'framer-motion';
import { 
  ArrowRight, Code2, Brain, Database, Globe, 
  Zap, Target, Award, TrendingUp, Sparkles,
  ChevronDown, Star, GitBranch, Coffee
} from 'lucide-react';
import './HomePage.css';

interface HomePageProps {
  onNavigate: (page: string) => void;
}

const HomePage: React.FC<HomePageProps> = ({ onNavigate }) => {
  const [typedText, setTypedText] = useState('');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const heroRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const isStatsInView = useInView(statsRef, { once: true });
  
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"]
  });
  
  const y = useTransform(scrollYProgress, [0, 1], ['0%', '50%']);
  const opacity = useTransform(scrollYProgress, [0, 1], [1, 0]);
  
  const fullText = "Data Science Innovator";
  
  // Counter animation for stats
  const [counters, setCounters] = useState({
    projects: 0,
    skills: 0,
    experience: 0,
    commits: 0
  });

  useEffect(() => {
    // Typing effect
    let index = 0;
    const typingInterval = setInterval(() => {
      if (index <= fullText.length) {
        setTypedText(fullText.slice(0, index));
        index++;
      } else {
        clearInterval(typingInterval);
      }
    }, 100);
    
    return () => clearInterval(typingInterval);
  }, []);

  useEffect(() => {
    // Mouse tracking for parallax
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 2,
        y: (e.clientY / window.innerHeight - 0.5) * 2,
      });
    };
    
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    // Animate counters when in view
    if (isStatsInView) {
      const animateCounter = (target: number, key: keyof typeof counters) => {
        const duration = 2000;
        const step = target / (duration / 16);
        let current = 0;
        
        const interval = setInterval(() => {
          current += step;
          if (current >= target) {
            current = target;
            clearInterval(interval);
          }
          setCounters(prev => ({ ...prev, [key]: Math.floor(current) }));
        }, 16);
      };
      
      animateCounter(15, 'projects');
      animateCounter(25, 'skills');
      animateCounter(3, 'experience');
      animateCounter(500, 'commits');
    }
  }, [isStatsInView]);

  const features = [
    {
      icon: <Brain />,
      title: 'Machine Learning',
      description: 'Building intelligent systems with PyTorch, TensorFlow, and cutting-edge algorithms',
      gradient: 'linear-gradient(135deg, #667eea, #764ba2)'
    },
    {
      icon: <Database />,
      title: 'Data Engineering',
      description: 'Processing massive datasets with SQL, Spark, and modern data pipelines',
      gradient: 'linear-gradient(135deg, #00d4aa, #00b894)'
    },
    {
      icon: <Globe />,
      title: 'Full-Stack Development',
      description: 'Creating stunning web applications with React, Node.js, and cloud technologies',
      gradient: 'linear-gradient(135deg, #f093fb, #f5576c)'
    },
    {
      icon: <Zap />,
      title: 'Performance Optimization',
      description: 'Achieving sub-100ms response times and 99.9% uptime in production systems',
      gradient: 'linear-gradient(135deg, #ffd700, #ffb800)'
    }
  ];

  const skills = [
    { name: 'Python', level: 95, category: 'Language' },
    { name: 'React', level: 90, category: 'Frontend' },
    { name: 'Machine Learning', level: 85, category: 'AI/ML' },
    { name: 'SQL', level: 88, category: 'Database' },
    { name: 'Node.js', level: 85, category: 'Backend' },
    { name: 'AWS', level: 80, category: 'Cloud' }
  ];

  return (
    <div className="home-page">
      {/* Hero Section */}
      <section ref={heroRef} className="hero-section">
        {/* Animated Background */}
        <div className="hero-background">
          <motion.div 
            className="bg-gradient-1"
            animate={{
              scale: [1, 1.2, 1],
              rotate: [0, 90, 0],
            }}
            transition={{ duration: 20, repeat: Infinity }}
          />
          <motion.div 
            className="bg-gradient-2"
            animate={{
              scale: [1.2, 1, 1.2],
              rotate: [90, 0, 90],
            }}
            transition={{ duration: 15, repeat: Infinity }}
          />
          <motion.div 
            className="bg-gradient-3"
            animate={{
              scale: [1, 1.3, 1],
              rotate: [0, -90, 0],
            }}
            transition={{ duration: 25, repeat: Infinity }}
          />
          
          {/* Floating particles */}
          <div className="particles">
            {[...Array(20)].map((_, i) => (
              <motion.div
                key={i}
                className="particle"
                initial={{ 
                  x: Math.random() * window.innerWidth,
                  y: Math.random() * window.innerHeight 
                }}
                animate={{
                  x: Math.random() * window.innerWidth,
                  y: Math.random() * window.innerHeight,
                }}
                transition={{
                  duration: Math.random() * 20 + 10,
                  repeat: Infinity,
                  repeatType: 'reverse',
                }}
              />
            ))}
          </div>
        </div>

        <motion.div 
          className="hero-content"
          style={{ y, opacity }}
        >
          {/* Animated Avatar */}
          <motion.div 
            className="hero-avatar"
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ 
              type: 'spring',
              stiffness: 260,
              damping: 20,
              duration: 0.8 
            }}
          >
            <motion.div 
              className="avatar-glow"
              animate={{ 
                scale: [1, 1.2, 1],
                opacity: [0.5, 0.8, 0.5] 
              }}
              transition={{ duration: 3, repeat: Infinity }}
            />
            <div className="avatar-content">
              <span className="avatar-text">JG</span>
            </div>
            <motion.div 
              className="avatar-ring ring-1"
              animate={{ rotate: 360 }}
              transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
            />
            <motion.div 
              className="avatar-ring ring-2"
              animate={{ rotate: -360 }}
              transition={{ duration: 15, repeat: Infinity, ease: 'linear' }}
            />
          </motion.div>

          {/* Hero Text */}
          <motion.h1 
            className="hero-title"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            Hi, I'm{' '}
            <span className="gradient-text">Joshua Gulizia</span>
          </motion.h1>

          <motion.div 
            className="hero-subtitle"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <span className="typed-text">{typedText}</span>
            <motion.span 
              className="cursor"
              animate={{ opacity: [1, 0, 1] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              |
            </motion.span>
          </motion.div>

          <motion.p 
            className="hero-description"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            Transforming complex data into actionable insights and building 
            intelligent systems that make real-world impact. Currently pursuing 
            Computer Science at the University of Houston with aspirations 
            for a PhD in Data Science.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div 
            className="hero-cta"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.9 }}
          >
            <motion.button 
              className="cta-primary"
              onClick={() => onNavigate('projects')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Sparkles size={20} />
              Explore My Work
              <ArrowRight size={20} />
            </motion.button>
            
            <motion.button 
              className="cta-secondary"
              onClick={() => onNavigate('contact')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Let's Connect
            </motion.button>
          </motion.div>
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div 
          className="scroll-indicator"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <span>Scroll to explore</span>
          <ChevronDown size={20} />
        </motion.div>
      </section>

      {/* Stats Section */}
      <section ref={statsRef} className="stats-section">
        <div className="container">
          <motion.div 
            className="stats-grid"
            initial={{ opacity: 0 }}
            animate={{ opacity: isStatsInView ? 1 : 0 }}
            transition={{ duration: 0.5 }}
          >
            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.05 }}
            >
              <div className="stat-icon">
                <Code2 size={32} />
              </div>
              <div className="stat-number">{counters.projects}+</div>
              <div className="stat-label">Projects Completed</div>
            </motion.div>

            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.05 }}
            >
              <div className="stat-icon">
                <Zap size={32} />
              </div>
              <div className="stat-number">{counters.skills}+</div>
              <div className="stat-label">Technologies</div>
            </motion.div>

            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.05 }}
            >
              <div className="stat-icon">
                <Coffee size={32} />
              </div>
              <div className="stat-number">{counters.experience}</div>
              <div className="stat-label">Years Experience</div>
            </motion.div>

            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.05 }}
            >
              <div className="stat-icon">
                <GitBranch size={32} />
              </div>
              <div className="stat-number">{counters.commits}+</div>
              <div className="stat-label">Git Commits</div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="section-title">What I Do</h2>
            <p className="section-subtitle">
              Specializing in cutting-edge technologies to deliver exceptional solutions
            </p>
          </motion.div>

          <div className="features-grid">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ 
                  scale: 1.05,
                  boxShadow: '0 20px 40px rgba(0,0,0,0.2)'
                }}
              >
                <motion.div 
                  className="feature-icon"
                  style={{ background: feature.gradient }}
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                >
                  {feature.icon}
                </motion.div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </motion.div>
            ))}
          </div>
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
            <h2 className="section-title">Technical Proficiency</h2>
            <p className="section-subtitle">
              My expertise across different technologies
            </p>
          </motion.div>

          <div className="skills-grid">
            {skills.map((skill, index) => (
              <motion.div
                key={index}
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
                <span className="skill-category">{skill.category}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <div className="container">
          <motion.div 
            className="cta-content"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <Star className="cta-icon" />
            <h2>Ready to Build Something Amazing?</h2>
            <p>Let's collaborate on your next big project</p>
            <motion.button 
              className="cta-button"
              onClick={() => onNavigate('contact')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Get In Touch
              <ArrowRight size={20} />
            </motion.button>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default HomePage;