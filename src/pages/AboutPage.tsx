// src/pages/AboutPage.tsx
import React from 'react';
import { motion } from 'framer-motion';
import { 
  GraduationCap, MapPin, Calendar, Award, 
  Target, Heart, Coffee, Code2, BookOpen,
  Sparkles, TrendingUp, Users, Star
} from 'lucide-react';
import './AboutPage.css';

const AboutPage: React.FC = () => {
  const education = {
    degree: 'Bachelor of Science in Computer Science',
    minor: 'Mathematics',
    university: 'University of Houston',
    location: 'Houston, TX',
    graduation: 'December 2026',
    gpa: '3.8/4.0',
    coursework: [
      'Machine Learning', 'Data Science', 'Database Systems', 
      'Data Structures', 'Statistics', 'Linear Algebra',
      'Algorithms', 'Artificial Intelligence'
    ]
  };

  const certifications = [
    {
      title: 'IBM Data Science Professional Certificate',
      issuer: 'IBM',
      date: '2024',
      skills: ['Machine Learning', 'SQL', 'Data Visualization']
    },
    {
      title: 'AWS Certified Cloud Practitioner',
      issuer: 'Amazon Web Services',
      date: '2024',
      skills: ['Cloud Computing', 'AWS Services', 'Security']
    }
  ];

  const interests = [
    { icon: <Code2 />, title: 'Coding', description: 'Building innovative solutions' },
    { icon: <TrendingUp />, title: 'Finance', description: 'Algorithmic trading & analysis' },
    { icon: <BookOpen />, title: 'Research', description: 'AI & Machine Learning papers' },
    { icon: <Users />, title: 'Mentoring', description: 'Teaching programming to students' }
  ];

  const values = [
    { icon: <Target />, title: 'Excellence', description: 'Striving for the highest quality in everything I do' },
    { icon: <Heart />, title: 'Passion', description: 'Genuinely loving the craft of building great software' },
    { icon: <Sparkles />, title: 'Innovation', description: 'Always looking for creative solutions to complex problems' },
    { icon: <Users />, title: 'Collaboration', description: 'Believing that great things are built by great teams' }
  ];

  return (
    <div className="about-page">
      {/* Hero Section */}
      <section className="about-hero">
        <motion.div 
          className="hero-content"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div 
            className="profile-image-container"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          >
            <div className="profile-image">
              <span>JG</span>
            </div>
            <motion.div 
              className="profile-ring"
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
            />
          </motion.div>
          
          <h1 className="hero-title">About Me</h1>
          <p className="hero-subtitle">
            Passionate about transforming data into actionable insights and building intelligent systems
          </p>
        </motion.div>
      </section>

      {/* Story Section */}
      <section className="story-section">
        <div className="container">
          <motion.div 
            className="story-content"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2>My Journey</h2>
            <div className="story-text">
              <p>
                I'm an aspiring data scientist with a deep passion for uncovering insights from complex datasets 
                and building intelligent systems that make real-world impact. Currently pursuing my Bachelor's 
                degree in Computer Science with a minor in Mathematics at the University of Houston, I've dedicated 
                myself to mastering the intersection of technology and data.
              </p>
              <p>
                My journey began with a fascination for how data can tell stories and predict the future. This led 
                me to dive deep into machine learning, statistical modeling, and full-stack development. Through 
                internships, personal projects, and continuous learning, I've built expertise in Python, React, 
                SQL, and various ML frameworks.
              </p>
              <p>
                My ultimate goal is to pursue a PhD in Data Science, where I plan to conduct cutting-edge research 
                in machine learning and artificial intelligence. I'm particularly interested in developing novel 
                algorithms that can extract meaningful patterns from massive, unstructured datasets.
              </p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Education Section */}
      <section className="education-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Education</h2>
          </motion.div>
          
          <motion.div 
            className="education-card"
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <div className="edu-header">
              <GraduationCap className="edu-icon" />
              <div className="edu-info">
                <h3>{education.degree}</h3>
                <p className="edu-minor">Minor in {education.minor}</p>
                <p className="edu-university">{education.university}</p>
              </div>
              <div className="edu-meta">
                <div className="meta-item">
                  <MapPin size={16} />
                  <span>{education.location}</span>
                </div>
                <div className="meta-item">
                  <Calendar size={16} />
                  <span>{education.graduation}</span>
                </div>
                <div className="meta-item">
                  <Award size={16} />
                  <span>GPA: {education.gpa}</span>
                </div>
              </div>
            </div>
            
            <div className="coursework">
              <h4>Relevant Coursework</h4>
              <div className="course-grid">
                {education.coursework.map((course, index) => (
                  <motion.span 
                    key={course}
                    className="course-chip"
                    initial={{ opacity: 0, scale: 0 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: index * 0.05 }}
                  >
                    {course}
                  </motion.span>
                ))}
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Certifications Section */}
      <section className="certifications-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Certifications</h2>
          </motion.div>
          
          <div className="certifications-grid">
            {certifications.map((cert, index) => (
              <motion.div 
                key={cert.title}
                className="cert-card"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05 }}
              >
                <Award className="cert-icon" />
                <h3>{cert.title}</h3>
                <p className="cert-issuer">{cert.issuer}</p>
                <p className="cert-date">{cert.date}</p>
                <div className="cert-skills">
                  {cert.skills.map(skill => (
                    <span key={skill} className="skill-tag">{skill}</span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="values-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>My Values</h2>
            <p>The principles that guide my work and life</p>
          </motion.div>
          
          <div className="values-grid">
            {values.map((value, index) => (
              <motion.div 
                key={value.title}
                className="value-card"
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <motion.div 
                  className="value-icon"
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.5 }}
                >
                  {value.icon}
                </motion.div>
                <h3>{value.title}</h3>
                <p>{value.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Interests Section */}
      <section className="interests-section">
        <div className="container">
          <motion.div 
            className="section-header"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2>Beyond Code</h2>
            <p>What drives me outside of work</p>
          </motion.div>
          
          <div className="interests-grid">
            {interests.map((interest, index) => (
              <motion.div 
                key={interest.title}
                className="interest-card"
                initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <div className="interest-icon">{interest.icon}</div>
                <div className="interest-content">
                  <h4>{interest.title}</h4>
                  <p>{interest.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Fun Facts */}
      <section className="fun-facts">
        <div className="container">
          <motion.div 
            className="facts-content"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h2>Fun Facts</h2>
            <div className="facts-grid">
              <motion.div 
                className="fact-item"
                whileHover={{ scale: 1.1 }}
              >
                <Coffee size={32} />
                <span className="fact-number">1000+</span>
                <span className="fact-label">Cups of Coffee</span>
              </motion.div>
              <motion.div 
                className="fact-item"
                whileHover={{ scale: 1.1 }}
              >
                <Code2 size={32} />
                <span className="fact-number">100K+</span>
                <span className="fact-label">Lines of Code</span>
              </motion.div>
              <motion.div 
                className="fact-item"
                whileHover={{ scale: 1.1 }}
              >
                <Star size={32} />
                <span className="fact-number">15+</span>
                <span className="fact-label">Projects Completed</span>
              </motion.div>
              <motion.div 
                className="fact-item"
                whileHover={{ scale: 1.1 }}
              >
                <Users size={32} />
                <span className="fact-number">200+</span>
                <span className="fact-label">Students Mentored</span>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default AboutPage;