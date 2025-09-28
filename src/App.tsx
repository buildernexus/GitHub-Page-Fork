import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home,
  User,
  Briefcase,
  Code2,
  Brain,
  Mail,
  Github,
  Linkedin,
  Download,
  ArrowRight,
  Menu,
  X,
  MapPin,
  Calendar,
  Award,
  TrendingUp,
  Database,
  Server,
  Globe,
  Terminal,
  ChevronRight,
  Sparkles,
  Zap,
  Target,
  Layers,
  Activity,
  BarChart3,
  GitBranch,
  CheckCircle,
  ExternalLink,
  Send,
  BookOpen,
  GraduationCap
} from 'lucide-react';

const App = () => {
  const [currentPage, setCurrentPage] = useState('home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [typedText, setTypedText] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  const fullText = "Data Science Innovator";

  useEffect(() => {
    // Initial loading
    setTimeout(() => setIsLoading(false), 1500);
    
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  useEffect(() => {
    if (!isLoading) {
      let index = 0;
      const interval = setInterval(() => {
        if (index <= fullText.length) {
          setTypedText(fullText.slice(0, index));
          index++;
        } else {
          clearInterval(interval);
        }
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isLoading]);

  const styles = {
    app: {
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #0a0a0a 0%, #0f0f0f 50%, #050505 100%)',
      color: '#ffffff',
      fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      position: 'relative',
      overflow: 'hidden',
    },
    backgroundPattern: {
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      opacity: 0.03,
      backgroundImage: `
        radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(255, 255, 255, 0.05) 0%, transparent 50%)
      `,
      pointerEvents: 'none',
      transition: 'all 0.3s ease',
      zIndex: 1,
    },
    navbar: {
      position: 'fixed' as const,
      top: 0,
      left: 0,
      right: 0,
      padding: '1.5rem 2rem',
      background: scrolled ? 'rgba(10, 10, 10, 0.95)' : 'transparent',
      backdropFilter: scrolled ? 'blur(20px)' : 'none',
      borderBottom: scrolled ? '1px solid rgba(255, 255, 255, 0.1)' : 'none',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      zIndex: 1000,
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    logo: {
      fontSize: '1.8rem',
      fontWeight: '800',
      letterSpacing: '-0.02em',
      background: 'linear-gradient(135deg, #ffffff 0%, #808080 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      cursor: 'pointer',
    },
    navMenu: {
      display: 'flex',
      gap: '2rem',
      alignItems: 'center',
    },
    navItem: {
      color: 'rgba(255, 255, 255, 0.7)',
      textDecoration: 'none',
      fontSize: '1rem',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      position: 'relative' as const,
      padding: '0.5rem 0',
    },
    navItemActive: {
      color: '#ffffff',
      fontWeight: '600',
    },
    mobileMenuButton: {
      display: 'none',
      background: 'none',
      border: 'none',
      color: '#ffffff',
      cursor: 'pointer',
      padding: '0.5rem',
    },
    content: {
      position: 'relative',
      zIndex: 2,
      minHeight: '100vh',
    },
    pageContainer: {
      paddingTop: '100px',
      paddingBottom: '4rem',
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '100px 2rem 4rem 2rem',
    },
    heroSection: {
      minHeight: 'calc(100vh - 100px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      textAlign: 'center' as const,
      position: 'relative' as const,
    },
    profileImage: {
      width: '200px',
      height: '200px',
      borderRadius: '50%',
      background: 'linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%)',
      border: '3px solid rgba(255, 255, 255, 0.1)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontSize: '4rem',
      fontWeight: '700',
      color: 'rgba(255, 255, 255, 0.9)',
      marginBottom: '2rem',
      boxShadow: '0 20px 40px rgba(0, 0, 0, 0.5)',
    },
    heroTitle: {
      fontSize: 'clamp(3rem, 8vw, 5rem)',
      fontWeight: '900',
      lineHeight: '1.1',
      background: 'linear-gradient(135deg, #ffffff 0%, #606060 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      marginBottom: '1rem',
      letterSpacing: '-0.03em',
    },
    heroSubtitle: {
      fontSize: 'clamp(1.5rem, 3vw, 2rem)',
      fontWeight: '300',
      color: 'rgba(255, 255, 255, 0.6)',
      marginBottom: '3rem',
    },
    ctaContainer: {
      display: 'flex',
      gap: '1.5rem',
      justifyContent: 'center',
      flexWrap: 'wrap',
    },
    primaryButton: {
      padding: '1rem 2.5rem',
      fontSize: '1.1rem',
      fontWeight: '600',
      background: 'linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%)',
      color: '#000000',
      border: 'none',
      borderRadius: '12px',
      cursor: 'pointer',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      boxShadow: '0 4px 15px rgba(255, 255, 255, 0.1)',
    },
    secondaryButton: {
      padding: '1rem 2.5rem',
      fontSize: '1.1rem',
      fontWeight: '600',
      background: 'transparent',
      color: '#ffffff',
      border: '2px solid rgba(255, 255, 255, 0.3)',
      borderRadius: '12px',
      cursor: 'pointer',
      transition: 'all 0.3s ease',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
    },
    statsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
      gap: '1.5rem',
      marginTop: '4rem',
      maxWidth: '600px',
    },
    statCard: {
      padding: '1.5rem',
      background: 'rgba(255, 255, 255, 0.03)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '16px',
      backdropFilter: 'blur(10px)',
      textAlign: 'center',
    },
    statValue: {
      fontSize: '2rem',
      fontWeight: '700',
      color: '#ffffff',
      marginBottom: '0.5rem',
    },
    statLabel: {
      fontSize: '0.9rem',
      color: 'rgba(255, 255, 255, 0.5)',
    },
    pageHeader: {
      textAlign: 'center',
      marginBottom: '4rem',
    },
    pageTitle: {
      fontSize: 'clamp(2.5rem, 5vw, 3.5rem)',
      fontWeight: '800',
      background: 'linear-gradient(135deg, #ffffff 0%, #808080 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent',
      marginBottom: '1rem',
    },
    pageSubtitle: {
      fontSize: '1.25rem',
      color: 'rgba(255, 255, 255, 0.6)',
      maxWidth: '600px',
      margin: '0 auto',
    },
    card: {
      background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)',
      border: '1px solid rgba(255, 255, 255, 0.1)',
      borderRadius: '20px',
      padding: '2rem',
      backdropFilter: 'blur(20px)',
      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      marginBottom: '2rem',
    },
    cardHover: {
      transform: 'translateY(-5px)',
      boxShadow: '0 20px 40px rgba(255, 255, 255, 0.05)',
      borderColor: 'rgba(255, 255, 255, 0.2)',
    },
    sectionTitle: {
      fontSize: '1.8rem',
      fontWeight: '700',
      marginBottom: '1.5rem',
      color: '#ffffff',
    },
    chip: {
      display: 'inline-block',
      padding: '0.4rem 1rem',
      background: 'rgba(255, 255, 255, 0.1)',
      border: '1px solid rgba(255, 255, 255, 0.2)',
      borderRadius: '20px',
      fontSize: '0.9rem',
      marginRight: '0.5rem',
      marginBottom: '0.5rem',
      transition: 'all 0.3s ease',
    },
    chipHover: {
      background: 'rgba(255, 255, 255, 0.15)',
      borderColor: 'rgba(255, 255, 255, 0.3)',
    },
    grid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '2rem',
    },
    timeline: {
      position: 'relative',
      paddingLeft: '3rem',
    },
    timelineItem: {
      position: 'relative',
      marginBottom: '3rem',
    },
    timelineDot: {
      position: 'absolute',
      left: '-3rem',
      top: '0.5rem',
      width: '40px',
      height: '40px',
      borderRadius: '50%',
      background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.05))',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      border: '2px solid rgba(255, 255, 255, 0.2)',
    },
    timelineLine: {
      position: 'absolute',
      left: '-1.5rem',
      top: '3rem',
      bottom: '-3rem',
      width: '2px',
      background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.2), transparent)',
    },
    mobileMenu: {
      position: 'fixed' as const,
      top: 0,
      right: mobileMenuOpen ? 0 : '-100%',
      width: '80%',
      maxWidth: '400px',
      height: '100vh',
      background: 'linear-gradient(180deg, #0a0a0a 0%, #1a1a1a 100%)',
      padding: '2rem',
      transition: 'right 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
      zIndex: 999,
      boxShadow: mobileMenuOpen ? '-10px 0 30px rgba(0, 0, 0, 0.5)' : 'none',
    },
    overlay: {
      position: 'fixed' as const,
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.5)',
      display: mobileMenuOpen ? 'block' : 'none',
      zIndex: 998,
    },
  };

  const navItems = [
    { id: 'home', label: 'Home', icon: <Home size={18} /> },
    { id: 'about', label: 'About', icon: <User size={18} /> },
    { id: 'experience', label: 'Experience', icon: <Briefcase size={18} /> },
    { id: 'projects', label: 'Projects', icon: <Code2 size={18} /> },
    { id: 'research', label: 'Research', icon: <Brain size={18} /> },
    { id: 'contact', label: 'Contact', icon: <Mail size={18} /> },
  ];

  const Navigation = () => (
    <>
      <nav style={styles.navbar}>
        <div style={styles.logo} onClick={() => setCurrentPage('home')}>JG</div>
        
        {/* Desktop Menu */}
        <div style={{ ...styles.navMenu, display: window.innerWidth > 768 ? 'flex' : 'none' }}>
          {navItems.map(item => (
            <div
              key={item.id}
              style={{
                ...styles.navItem,
                ...(currentPage === item.id ? styles.navItemActive : {})
              }}
              onClick={() => setCurrentPage(item.id)}
              onMouseEnter={(e) => (e.target as HTMLElement).style.color = '#ffffff'}
              onMouseLeave={(e) => (e.target as HTMLElement).style.color = currentPage === item.id ? '#ffffff' : 'rgba(255, 255, 255, 0.7)'}
            >
              {item.label}
            </div>
          ))}
        </div>
        
        {/* Mobile Menu Button */}
        <button
          style={{ ...styles.mobileMenuButton, display: window.innerWidth <= 768 ? 'block' : 'none' }}
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </nav>
      
      {/* Mobile Menu */}
      <div style={styles.overlay} onClick={() => setMobileMenuOpen(false)} />
      <div style={styles.mobileMenu}>
        <div style={{ marginBottom: '2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={styles.logo}>JG</div>
          <button
            style={{ background: 'none', border: 'none', color: '#ffffff', cursor: 'pointer' }}
            onClick={() => setMobileMenuOpen(false)}
          >
            <X size={24} />
          </button>
        </div>
        {navItems.map(item => (
          <div
            key={item.id}
            style={{
              padding: '1rem',
              borderRadius: '12px',
              background: currentPage === item.id ? 'rgba(255, 255, 255, 0.1)' : 'transparent',
              marginBottom: '0.5rem',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              display: 'flex',
              alignItems: 'center',
              gap: '1rem',
            }}
            onClick={() => {
              setCurrentPage(item.id);
              setMobileMenuOpen(false);
            }}
          >
            {item.icon}
            <span style={{ fontSize: '1.1rem' }}>{item.label}</span>
          </div>
        ))}
      </div>
    </>
  );

  const HomePage = () => {
    const [hoveredStat, setHoveredStat] = useState(null);
    
    return (
      <div style={styles.heroSection}>
        {/* Floating particles animation */}
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          overflow: 'hidden',
          pointerEvents: 'none',
        }}>
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              style={{
                position: 'absolute',
                width: '2px',
                height: '2px',
                background: 'rgba(255, 255, 255, 0.3)',
                borderRadius: '50%',
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                y: [-20, 20],
                x: [-20, 20],
                opacity: [0.2, 0.5, 0.2],
              }}
              transition={{
                duration: 3 + Math.random() * 3,
                repeat: Infinity,
                repeatType: 'reverse',
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8 }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <motion.div
              style={{
                ...styles.profileImage,
                position: 'relative',
              }}
              whileHover={{ scale: 1.05, rotate: 5 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <div style={{
                position: 'absolute',
                top: '-4px',
                left: '-4px',
                right: '-4px',
                bottom: '-4px',
                borderRadius: '50%',
                background: 'linear-gradient(45deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))',
                animation: 'rotate 3s linear infinite',
              }} />
              JG
            </motion.div>
            
            <motion.h1 
              style={styles.heroTitle}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.8 }}
            >
              Joshua Gulizia
            </motion.h1>
            <motion.div 
              style={styles.heroSubtitle}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              {typedText}
              <span style={{ 
                opacity: typedText === fullText ? 0 : 1,
                animation: 'blink 1s infinite' 
              }}>|</span>
            </motion.div>
            
            <motion.div 
              style={{...styles.ctaContainer, flexWrap: 'wrap' as const}}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <motion.button
                style={styles.primaryButton}
                whileHover={{ 
                  transform: 'translateY(-2px)', 
                  boxShadow: '0 10px 30px rgba(255, 255, 255, 0.2)' 
                }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setCurrentPage('about')}
              >
                Explore My Work <ArrowRight size={20} />
              </motion.button>
              <motion.button
                style={styles.secondaryButton}
                whileHover={{ 
                  background: 'rgba(255, 255, 255, 0.05)',
                  borderColor: 'rgba(255, 255, 255, 0.5)' 
                }}
                whileTap={{ scale: 0.98 }}
              >
                <Download size={20} /> Download Resume
              </motion.button>
            </motion.div>
            
            <div style={styles.statsGrid}>
              {[
                { value: '15+', label: 'Projects', icon: <Code2 size={24} /> },
                { value: '3', label: 'Years Experience', icon: <Briefcase size={24} /> },
                { value: '20+', label: 'Skills', icon: <Zap size={24} /> },
              ].map((stat, index) => (
                <motion.div
                  key={stat.label}
                  style={{
                    ...styles.statCard,
                    transform: hoveredStat === index ? 'scale(1.05)' : 'scale(1)',
                    boxShadow: hoveredStat === index ? '0 10px 30px rgba(255, 255, 255, 0.1)' : 'none',
                    textAlign: 'center' as const,
                  }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 + index * 0.1 }}
                  onMouseEnter={() => setHoveredStat(index)}
                  onMouseLeave={() => setHoveredStat(null)}
                >
                  <motion.div
                    style={{ 
                      color: 'rgba(255, 255, 255, 0.4)', 
                      marginBottom: '0.5rem',
                      display: 'flex',
                      justifyContent: 'center',
                    }}
                    animate={{ 
                      rotate: hoveredStat === index ? 360 : 0,
                    }}
                    transition={{ duration: 0.5 }}
                  >
                    {stat.icon}
                  </motion.div>
                  <div style={styles.statValue}>{stat.value}</div>
                  <div style={styles.statLabel}>{stat.label}</div>
                </motion.div>
              ))}
            </div>

            {/* Scroll indicator */}
            <motion.div
              style={{
                position: 'absolute',
                bottom: '2rem',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.5rem',
              }}
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span style={{ fontSize: '0.8rem', color: 'rgba(255, 255, 255, 0.4)' }}>Scroll to explore</span>
              <ChevronRight size={20} style={{ transform: 'rotate(90deg)', color: 'rgba(255, 255, 255, 0.4)' }} />
            </motion.div>
          </div>
        </motion.div>
      </div>
    );
  };

  const AboutPage = () => (
    <div style={styles.pageContainer}>
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div style={{...styles.pageHeader, textAlign: 'center' as const}}>
          <h1 style={styles.pageTitle}>About Me</h1>
          <p style={styles.pageSubtitle}>
            Passionate about transforming data into actionable insights
          </p>
        </div>
        
        <div style={styles.grid}>
          <div style={{ gridColumn: window.innerWidth > 768 ? 'span 2' : 'span 1' }}>
            <div style={styles.card}>
              <h2 style={styles.sectionTitle}>My Journey</h2>
              <p style={{ fontSize: '1.1rem', lineHeight: '1.8', color: 'rgba(255, 255, 255, 0.8)', marginBottom: '1.5rem' }}>
                I am an aspiring data scientist with a passion for uncovering insights from complex datasets 
                and building intelligent systems that make real-world impact. Currently pursuing my Bachelor's 
                degree in Computer Science with a minor in Mathematics at the University of Houston.
              </p>
              <p style={{ fontSize: '1.1rem', lineHeight: '1.8', color: 'rgba(255, 255, 255, 0.8)' }}>
                My ultimate goal is to pursue a PhD in Data Science, where I plan to conduct cutting-edge 
                research in machine learning, statistical modeling, and artificial intelligence.
              </p>
            </div>
            
            <div style={styles.card}>
              <h2 style={styles.sectionTitle}>Technical Expertise</h2>
              <div style={styles.grid}>
                {[
                  { category: 'Languages', skills: ['Python', 'R', 'JavaScript', 'SQL', 'C++'] },
                  { category: 'ML/AI', skills: ['PyTorch', 'Scikit-learn', 'TensorFlow', 'Keras'] },
                  { category: 'Data', skills: ['Pandas', 'NumPy', 'Spark', 'Hadoop'] },
                  { category: 'Tools', skills: ['Docker', 'Git', 'AWS', 'Power BI'] },
                ].map(group => (
                  <div key={group.category}>
                    <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem', color: 'rgba(255, 255, 255, 0.6)' }}>
                      {group.category}
                    </h3>
                    <div>
                      {group.skills.map(skill => (
                        <span
                          key={skill}
                          style={styles.chip}
                          onMouseEnter={(e) => {
                            (e.target as HTMLElement).style.background = 'rgba(255, 255, 255, 0.15)';
                            (e.target as HTMLElement).style.borderColor = 'rgba(255, 255, 255, 0.3)';
                          }}
                          onMouseLeave={(e) => {
                            (e.target as HTMLElement).style.background = 'rgba(255, 255, 255, 0.1)';
                            (e.target as HTMLElement).style.borderColor = 'rgba(255, 255, 255, 0.2)';
                          }}
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          <div>
            <div style={styles.card}>
              <h2 style={{ ...styles.sectionTitle, fontSize: '1.4rem' }}>Quick Facts</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <GraduationCap size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                  <div>
                    <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>Education</div>
                    <div>University of Houston</div>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <MapPin size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                  <div>
                    <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>Location</div>
                    <div>Houston, TX</div>
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <Calendar size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                  <div>
                    <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>Graduation</div>
                    <div>May 2026</div>
                  </div>
                </div>
              </div>
            </div>
            
            <div style={styles.card}>
              <h2 style={{ ...styles.sectionTitle, fontSize: '1.4rem' }}>Research Interests</h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                {['Machine Learning', 'Quantitative Analytics', 'Statistical Modeling', 'Predictive Analytics'].map(interest => (
                  <div key={interest} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <ChevronRight size={16} style={{ color: 'rgba(255, 255, 255, 0.4)' }} />
                    <span>{interest}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );

  const ExperiencePage = () => {
    const experiences = [
      {
        title: 'Data Science Intern',
        company: 'Sun & Ski Sports',
        location: 'Houston, TX',
        date: 'Aug 2025 - Present',
        description: [
          'Analyzed 10M+ rows of sales data using PostgreSQL and pandas across 40 retail locations',
          'Created 5 Power BI dashboards for performance metrics, reducing weekly reporting time by 80%',
          'Streamlined data cleaning using Python, resolving 1,700+ data inconsistencies',
        ],
        icon: <Database size={20} />,
      },
      {
        title: 'Operations Officer',
        company: 'Code Coogs',
        location: 'Houston, TX',
        date: 'Jan 2025 - Present',
        description: [
          'Conduct technical workshops teaching full-stack development to 200+ students per semester',
          'Collaborate with officers to coordinate workshop scheduling and curriculum development',
          'Mentor students on programming fundamentals and industry best practices',
        ],
        icon: <Code2 size={20} />,
      },
      {
        title: 'QuickBooks Bookkeeper',
        company: 'Reel It Inn Rentals',
        location: 'Freeport, TX',
        date: 'Apr 2023 - Jul 2025',
        description: [
          'Managed $60K yearly revenue processing 50+ monthly transactions in QuickBooks',
          'Maintained financial records with 99% accuracy for accounts payable and receivable',
          'Generated monthly financial reports to track revenue trends',
        ],
        icon: <TrendingUp size={20} />,
      },
    ];

    return (
      <div style={styles.pageContainer}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div style={{...styles.pageHeader, textAlign: 'center' as const}}>
            <h1 style={styles.pageTitle}>Experience</h1>
            <p style={styles.pageSubtitle}>Professional journey and achievements</p>
          </div>
          
          <div style={{...styles.timeline, position: 'relative' as const}}>
            {experiences.map((exp, index) => (
              <motion.div
                key={index}
                style={{...styles.timelineItem, position: 'relative' as const}}
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
              >
                <div style={{...styles.timelineDot, position: 'absolute' as const}}>
                  {exp.icon}
                </div>
                {index < experiences.length - 1 && <div style={{...styles.timelineLine, position: 'absolute' as const}} />}
                
                <div 
                  style={styles.card}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.transform = 'translateY(-5px)';
                    e.currentTarget.style.boxShadow = '0 20px 40px rgba(255, 255, 255, 0.05)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }}
                >
                  <h3 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{exp.title}</h3>
                  <p style={{ color: 'rgba(255, 255, 255, 0.6)', marginBottom: '0.5rem' }}>
                    {exp.company} • {exp.location}
                  </p>
                  <p style={{ color: 'rgba(255, 255, 255, 0.5)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
                    {exp.date}
                  </p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {exp.description.map((item, i) => (
                      <div key={i} style={{ display: 'flex', gap: '0.8rem' }}>
                        <CheckCircle size={20} style={{ color: 'rgba(255, 255, 255, 0.4)', flexShrink: 0, marginTop: '2px' }} />
                        <span style={{ color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6' }}>{item}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    );
  };

  const ProjectsPage = () => {
    const [hoveredProject, setHoveredProject] = useState(null);
    
    const projects = [
      {
        title: 'ML-Powered Algorithmic Trading Platform',
        description: 'Built ML pipeline with LSTM and ensemble models processing 500K+ daily volume stocks. Developed distributable trading tool with 60% confidence thresholds.',
        tech: ['Python', 'PyTorch', 'React', 'MySQL', 'Docker'],
        status: 'In Development',
        featured: true,
        metrics: ['500K+ Daily Volume', '60% Confidence', 'Real-time Processing'],
        gradient: 'linear-gradient(135deg, rgba(255, 215, 0, 0.05), rgba(255, 165, 0, 0.02))',
      },
      {
        title: 'StockScope - Real-Time Stock Monitoring',
        description: 'Stock screening application tracking 500+ stocks with real-time price updates and WebSocket data pipeline processing 500 API calls per second.',
        tech: ['React', 'Node.js', 'PostgreSQL', 'AWS'],
        status: 'Completed',
        github: 'https://github.com/jguliz/Stock-Screener',
        metrics: ['500+ Stocks', '< 100ms Response', '500 API/sec'],
      },
      {
        title: 'Cougar Library Management System',
        description: 'Library management platform with role-based authentication for 3 user types, inventory system, and automated fine calculations.',
        tech: ['React', 'Node.js', 'MySQL', 'Express.js'],
        status: 'Completed',
        github: 'https://github.com/jguliz/Library_Management_System',
        metrics: ['3 User Roles', 'Auto Fine Calc', 'Real-time Inventory'],
      },
      {
        title: 'Personal Portfolio Website',
        description: 'Modern, responsive portfolio website with dynamic animations and professional design showcasing projects and experience.',
        tech: ['React', 'TypeScript', 'Framer Motion'],
        status: 'Completed',
        github: 'https://github.com/jguliz/GitHub-Page',
        metrics: ['100% Responsive', 'Dynamic Animations', 'Modern Design'],
      },
    ];

    const featuredProject = projects.find(p => p.featured);
    const otherProjects = projects.filter(p => !p.featured);

    return (
      <div style={styles.pageContainer}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div style={{...styles.pageHeader, textAlign: 'center' as const}}>
            <motion.h1 
              style={styles.pageTitle}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              Projects
            </motion.h1>
            <motion.p 
              style={styles.pageSubtitle}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              Building the future through innovative solutions
            </motion.p>
          </div>
          
          {featuredProject && (
            <motion.div
              style={{
                ...styles.card,
                background: featuredProject.gradient || 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%)',
                border: '2px solid rgba(255, 255, 255, 0.15)',
                padding: '3rem',
                marginBottom: '3rem',
                position: 'relative',
                overflow: 'hidden',
              }}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              whileHover={{ scale: 1.02 }}
            >
              {/* Animated background gradient */}
              <motion.div
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at 50% 50%, rgba(255, 215, 0, 0.05), transparent)',
                  opacity: 0,
                }}
                animate={{ opacity: [0, 0.5, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
              />
              
              <div style={{ position: 'relative', zIndex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                  <motion.div
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
                  >
                    <Sparkles size={24} style={{ color: '#ffd700' }} />
                  </motion.div>
                  <span style={{ 
                    padding: '0.4rem 1.2rem', 
                    background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.15), rgba(255, 165, 0, 0.1))', 
                    border: '1px solid rgba(255, 215, 0, 0.4)', 
                    borderRadius: '25px',
                    fontSize: '0.9rem',
                    color: '#ffd700',
                    fontWeight: '600',
                    letterSpacing: '0.5px',
                  }}>
                    ⭐ Featured Project
                  </span>
                </div>
                
                <h2 style={{ fontSize: '2.2rem', marginBottom: '1rem', fontWeight: '700' }}>{featuredProject.title}</h2>
                <p style={{ fontSize: '1.15rem', color: 'rgba(255, 255, 255, 0.85)', marginBottom: '2rem', lineHeight: '1.7' }}>
                  {featuredProject.description}
                </p>
                
                {/* Metrics */}
                <div style={{ display: 'flex', gap: '2rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                  {featuredProject.metrics.map(metric => (
                    <div key={metric} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Target size={16} style={{ color: '#ffd700' }} />
                      <span style={{ fontSize: '0.95rem', color: 'rgba(255, 255, 255, 0.7)', fontWeight: '500' }}>{metric}</span>
                    </div>
                  ))}
                </div>
                
                <div style={{ marginBottom: '1.5rem' }}>
                  {featuredProject.tech.map((tech, i) => (
                    <motion.span 
                      key={tech} 
                      style={{
                        ...styles.chip,
                        background: 'rgba(255, 215, 0, 0.1)',
                        borderColor: 'rgba(255, 215, 0, 0.3)',
                      }}
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.5 + i * 0.05 }}
                    >
                      {tech}
                    </motion.span>
                  ))}
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                  <Activity size={16} style={{ color: 'rgba(255, 255, 255, 0.5)' }} />
                  <span style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                    Status: <strong style={{ color: '#ffd700' }}>{featuredProject.status}</strong>
                  </span>
                </div>
              </div>
            </motion.div>
          )}
          
          <div style={styles.grid}>
            {otherProjects.map((project, index) => (
              <motion.div
                key={index}
                style={{
                  ...styles.card,
                  cursor: 'pointer',
                  position: 'relative',
                  overflow: 'hidden',
                  transform: hoveredProject === index ? 'translateY(-8px) scale(1.02)' : 'translateY(0) scale(1)',
                  boxShadow: hoveredProject === index ? '0 25px 50px rgba(255, 255, 255, 0.08)' : 'none',
                }}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                onMouseEnter={() => setHoveredProject(index)}
                onMouseLeave={() => setHoveredProject(null)}
              >
                {/* Hover gradient effect */}
                <motion.div
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '3px',
                    background: 'linear-gradient(90deg, #ffffff, rgba(255, 255, 255, 0.3))',
                    scaleX: hoveredProject === index ? 1 : 0,
                    transformOrigin: 'left',
                  }}
                  animate={{ scaleX: hoveredProject === index ? 1 : 0 }}
                  transition={{ duration: 0.3 }}
                />
                
                <h3 style={{ fontSize: '1.5rem', marginBottom: '1rem', fontWeight: '600' }}>{project.title}</h3>
                <p style={{ color: 'rgba(255, 255, 255, 0.75)', marginBottom: '1.5rem', lineHeight: '1.6' }}>
                  {project.description}
                </p>
                
                {/* Metrics */}
                <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
                  {project.metrics.map(metric => (
                    <div key={metric} style={{ 
                      fontSize: '0.8rem', 
                      color: 'rgba(255, 255, 255, 0.5)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.3rem'
                    }}>
                      <div style={{ width: '4px', height: '4px', borderRadius: '50%', background: 'rgba(255, 255, 255, 0.4)' }} />
                      {metric}
                    </div>
                  ))}
                </div>
                
                <div style={{ marginBottom: '1.5rem' }}>
                  {project.tech.map(tech => (
                    <span 
                      key={tech} 
                      style={{ 
                        ...styles.chip, 
                        fontSize: '0.85rem',
                        padding: '0.3rem 0.8rem',
                        background: hoveredProject === index ? 'rgba(255, 255, 255, 0.15)' : 'rgba(255, 255, 255, 0.1)',
                      }}
                    >
                      {tech}
                    </span>
                  ))}
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ 
                    color: 'rgba(255, 255, 255, 0.5)', 
                    fontSize: '0.9rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem'
                  }}>
                    {project.status === 'Completed' ? (
                      <CheckCircle size={16} style={{ color: '#66bb6a' }} />
                    ) : (
                      <Activity size={16} />
                    )}
                    {project.status}
                  </span>
                  {project.github && (
                    <motion.a 
                      href={project.github} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      style={{ 
                        color: 'rgba(255, 255, 255, 0.6)', 
                        textDecoration: 'none',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.3rem',
                      }}
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                      onClick={(e) => e.stopPropagation()}
                    >
                      <Github size={20} />
                      <ExternalLink size={14} />
                    </motion.a>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    );
  };

  const ResearchPage = () => {
    const researchAreas = [
      {
        title: 'Machine Learning & AI',
        icon: <Brain size={32} />,
        topics: ['Deep Learning', 'Natural Language Processing', 'Computer Vision', 'Reinforcement Learning'],
        color: 'rgba(255, 255, 255, 0.1)',
      },
      {
        title: 'Quantitative Analytics',
        icon: <BarChart3 size={32} />,
        topics: ['Statistical Modeling', 'Time Series Analysis', 'Risk Assessment', 'Financial Modeling'],
        color: 'rgba(255, 255, 255, 0.08)',
      },
      {
        title: 'Mathematical Foundations',
        icon: <Activity size={32} />,
        topics: ['Linear Algebra', 'Optimization Theory', 'Probability & Statistics', 'Algorithm Design'],
        color: 'rgba(255, 255, 255, 0.06)',
      },
    ];

    return (
      <div style={styles.pageContainer}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div style={{...styles.pageHeader, textAlign: 'center' as const}}>
            <h1 style={styles.pageTitle}>Research Interests</h1>
            <p style={styles.pageSubtitle}>Exploring the frontiers of data science and AI</p>
          </div>
          
          <div style={styles.grid}>
            {researchAreas.map((area, index) => (
              <motion.div
                key={index}
                style={{
                  ...styles.card,
                  background: area.color,
                }}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.transform = 'translateY(-10px)';
                  e.currentTarget.style.boxShadow = '0 30px 60px rgba(255, 255, 255, 0.08)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'translateY(0)';
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
                  <div style={{ color: 'rgba(255, 255, 255, 0.7)' }}>{area.icon}</div>
                  <h3 style={{ fontSize: '1.4rem' }}>{area.title}</h3>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  {area.topics.map(topic => (
                    <div key={topic} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <ChevronRight size={16} style={{ color: 'rgba(255, 255, 255, 0.5)' }} />
                      <span style={{ color: 'rgba(255, 255, 255, 0.85)' }}>{topic}</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
          
          <motion.div
            style={styles.card}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <h2 style={styles.sectionTitle}>Research Vision</h2>
            <p style={{ fontSize: '1.1rem', lineHeight: '1.8', color: 'rgba(255, 255, 255, 0.8)', marginBottom: '1.5rem' }}>
              My research vision focuses on the intersection of Machine Learning and Quantitative Analytics, 
              where I aim to develop novel methodologies that combine advanced statistical techniques with 
              cutting-edge ML algorithms to solve complex real-world problems.
            </p>
            <p style={{ fontSize: '1.1rem', lineHeight: '1.8', color: 'rgba(255, 255, 255, 0.8)' }}>
              I'm particularly interested in creating robust analytical frameworks that can extract meaningful 
              insights from large-scale datasets across various domains, with applications in finance, 
              healthcare, and technology.
            </p>
          </motion.div>
        </motion.div>
      </div>
    );
  };

  const ContactPage = () => {
    const [formData, setFormData] = useState({ name: '', email: '', message: '' });
    const [showSuccess, setShowSuccess] = useState(false);

    const handleSubmit = (e) => {
      e.preventDefault();
      setShowSuccess(true);
      setFormData({ name: '', email: '', message: '' });
      setTimeout(() => setShowSuccess(false), 5000);
    };

    return (
      <div style={styles.pageContainer}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div style={{...styles.pageHeader, textAlign: 'center' as const}}>
            <h1 style={styles.pageTitle}>Get In Touch</h1>
            <p style={styles.pageSubtitle}>Let's connect and create something amazing together</p>
          </div>
          
          <div style={{ ...styles.grid, maxWidth: '1000px', margin: '0 auto' }}>
            <div style={{ gridColumn: window.innerWidth > 768 ? 'span 2' : 'span 1' }}>
              <div style={styles.card}>
                <h2 style={styles.sectionTitle}>Send Me a Message</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'rgba(255, 255, 255, 0.7)' }}>
                      Your Name
                    </label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      style={{
                        width: '100%',
                        padding: '0.8rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                        color: '#ffffff',
                        fontSize: '1rem',
                        outline: 'none',
                        transition: 'all 0.3s ease',
                      }}
                      onFocus={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.5)'}
                      onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)'}
                    />
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'rgba(255, 255, 255, 0.7)' }}>
                      Your Email
                    </label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                      style={{
                        width: '100%',
                        padding: '0.8rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                        color: '#ffffff',
                        fontSize: '1rem',
                        outline: 'none',
                        transition: 'all 0.3s ease',
                      }}
                      onFocus={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.5)'}
                      onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)'}
                    />
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', marginBottom: '0.5rem', color: 'rgba(255, 255, 255, 0.7)' }}>
                      Message
                    </label>
                    <textarea
                      value={formData.message}
                      onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                      rows={5}
                      style={{
                        width: '100%',
                        padding: '0.8rem',
                        background: 'rgba(255, 255, 255, 0.05)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                        color: '#ffffff',
                        fontSize: '1rem',
                        outline: 'none',
                        transition: 'all 0.3s ease',
                        resize: 'vertical',
                      }}
                      onFocus={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.5)'}
                      onBlur={(e) => e.target.style.borderColor = 'rgba(255, 255, 255, 0.2)'}
                    />
                  </div>
                  
                  <motion.button
                    onClick={handleSubmit}
                    style={{
                      ...styles.primaryButton,
                      width: 'fit-content',
                    }}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    Send Message <Send size={20} />
                  </motion.button>
                  
                  {showSuccess && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      style={{
                        padding: '1rem',
                        background: 'rgba(102, 187, 106, 0.1)',
                        border: '1px solid rgba(102, 187, 106, 0.3)',
                        borderRadius: '8px',
                        color: '#66bb6a',
                      }}
                    >
                      Message sent successfully! I'll get back to you soon.
                    </motion.div>
                  )}
                </div>
              </div>
            </div>
            
            <div>
              <div style={styles.card}>
                <h2 style={{ ...styles.sectionTitle, fontSize: '1.4rem' }}>Contact Information</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Mail size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                    <div>
                      <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>Email</div>
                      <a 
                        href="mailto:jgulizia1205@gmail.com" 
                        style={{ color: '#ffffff', textDecoration: 'none' }}
                      >
                        jgulizia1205@gmail.com
                      </a>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Linkedin size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                    <div>
                      <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>LinkedIn</div>
                      <a 
                        href="https://linkedin.com/in/josh-gulizia" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={{ color: '#ffffff', textDecoration: 'none' }}
                      >
                        josh-gulizia
                      </a>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <Github size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                    <div>
                      <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>GitHub</div>
                      <a 
                        href="https://github.com/jguliz" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        style={{ color: '#ffffff', textDecoration: 'none' }}
                      >
                        jguliz
                      </a>
                    </div>
                  </div>
                  
                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <MapPin size={24} style={{ color: 'rgba(255, 255, 255, 0.6)' }} />
                    <div>
                      <div style={{ fontSize: '0.9rem', color: 'rgba(255, 255, 255, 0.5)' }}>Location</div>
                      <div>Houston, Texas</div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div style={{ ...styles.card, marginTop: '2rem' }}>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem' }}>Let's Connect</h3>
                <p style={{ color: 'rgba(255, 255, 255, 0.8)', lineHeight: '1.6' }}>
                  I'm always open to discussing data science, research opportunities, 
                  and potential collaborations. Feel free to reach out!
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    );
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'home':
        return <HomePage />;
      case 'about':
        return <AboutPage />;
      case 'experience':
        return <ExperiencePage />;
      case 'projects':
        return <ProjectsPage />;
      case 'research':
        return <ResearchPage />;
      case 'contact':
        return <ContactPage />;
      default:
        return <HomePage />;
    }
  };

  // Loading Screen
  if (isLoading) {
    return (
      <div style={{
        ...styles.app,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        position: 'relative' as const,
      }}>
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          style={{ textAlign: 'center' }}
        >
          <motion.div
            style={{
              width: '100px',
              height: '100px',
              borderRadius: '50%',
              border: '3px solid rgba(255, 255, 255, 0.1)',
              borderTop: '3px solid rgba(255, 255, 255, 0.5)',
              margin: '0 auto 2rem',
            }}
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
          />
          <motion.h2
            style={{
              fontSize: '2rem',
              fontWeight: '700',
              background: 'linear-gradient(135deg, #ffffff 0%, #808080 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              marginBottom: '1rem',
            }}
            animate={{ opacity: [0.5, 1, 0.5] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            Joshua Gulizia
          </motion.h2>
          <motion.p
            style={{ color: 'rgba(255, 255, 255, 0.5)' }}
            animate={{ opacity: [0, 1, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          >
            Loading Portfolio...
          </motion.p>
        </motion.div>
      </div>
    );
  }

  return (
    <div style={{...styles.app, position: 'relative' as const}}>
      <div style={{...styles.backgroundPattern, position: 'fixed' as const, pointerEvents: 'none' as const}} />
      <Navigation />
      <div style={{...styles.content, position: 'relative' as const}}>
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPage}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
            {renderPage()}
          </motion.div>
        </AnimatePresence>
      </div>
      
      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
        
        @keyframes rotate {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        @keyframes shimmer {
          0% { background-position: -1000px 0; }
          100% { background-position: 1000px 0; }
        }
        
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        
        body {
          overflow-x: hidden;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        
        ::selection {
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
        }
        
        ::-webkit-scrollbar {
          width: 12px;
          height: 12px;
        }
        
        ::-webkit-scrollbar-track {
          background: rgba(255, 255, 255, 0.02);
          border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
          border-radius: 10px;
          border: 2px solid transparent;
          background-clip: padding-box;
        }
        
        ::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(180deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2));
        }
        
        ::-webkit-scrollbar-corner {
          background: transparent;
        }
        
        input:focus, textarea:focus {
          outline: none;
        }
        
        a {
          transition: all 0.3s ease;
        }
        
        @media (max-width: 768px) {
          .grid {
            grid-template-columns: 1fr !important;
          }
          
          .page-container {
            padding: 80px 1rem 3rem 1rem !important;
          }
          
          .hero-title {
            font-size: 2.5rem !important;
          }
          
          .hero-subtitle {
            font-size: 1.2rem !important;
          }
          
          .card {
            padding: 1.5rem !important;
          }
          
          .stats-grid {
            grid-template-columns: repeat(3, 1fr) !important;
          }
        }
        
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }
        
        /* Smooth scrolling */
        html {
          scroll-behavior: smooth;
        }
        
        /* Loading animation */
        .loading-pulse {
          animation: pulse 2s ease-in-out infinite;
        }
        
        /* Glassmorphism effect */
        .glass {
          background: rgba(255, 255, 255, 0.03);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Gradient text animation */
        .gradient-text {
          background: linear-gradient(135deg, #ffffff, #808080, #ffffff);
          background-size: 200% 200%;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          animation: shimmer 3s linear infinite;
        }
        
        /* Hover lift effect */
        .hover-lift {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .hover-lift:hover {
          transform: translateY(-5px);
          box-shadow: 0 20px 40px rgba(255, 255, 255, 0.1);
        }
        
        /* Responsive text */
        @media (max-width: 640px) {
          h1 { font-size: 2rem !important; }
          h2 { font-size: 1.5rem !important; }
          h3 { font-size: 1.25rem !important; }
          p { font-size: 0.95rem !important; }
        }
        
        /* Print styles */
        @media print {
          body {
            background: white;
            color: black;
          }
          
          nav, .mobile-menu, .overlay {
            display: none !important;
          }
        }
      `}</style>
    </div>
  );
};

export default App;