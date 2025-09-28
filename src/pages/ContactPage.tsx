// src/pages/ContactPage.tsx
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Mail, Phone, MapPin, Github, Linkedin,
  Send, MessageSquare, Clock, Calendar,
  CheckCircle, AlertCircle, Sparkles,
  Download, ArrowRight, Twitter, Globe,
  Coffee, Users, Briefcase, Code2
} from 'lucide-react';
import './ContactPage.css';

interface FormData {
  name: string;
  email: string;
  subject: string;
  message: string;
  type: 'general' | 'collaboration' | 'opportunity' | 'research';
}

const ContactPage: React.FC = () => {
  const [formData, setFormData] = useState<FormData>({
    name: '',
    email: '',
    subject: '',
    message: '',
    type: 'general'
  });

  const [formStatus, setFormStatus] = useState<{
    type: 'idle' | 'loading' | 'success' | 'error';
    message: string;
  }>({
    type: 'idle',
    message: ''
  });

  const [focusedField, setFocusedField] = useState<string | null>(null);

  const contactInfo = {
    email: 'jgulizia1205@gmail.com',
    phone: '(832) 221-8301',
    location: 'Houston, Texas',
    availability: 'Available for opportunities',
    responseTime: 'Usually responds within 24 hours'
  };

  const socialLinks = [
    {
      name: 'GitHub',
      icon: <Github />,
      url: 'https://github.com/jguliz',
      username: '@jguliz',
      color: '#333'
    },
    {
      name: 'LinkedIn',
      icon: <Linkedin />,
      url: 'https://linkedin.com/in/josh-gulizia-401474290',
      username: 'Josh Gulizia',
      color: '#0077b5'
    },
    {
      name: 'Email',
      icon: <Mail />,
      url: 'mailto:jgulizia1205@gmail.com',
      username: 'jgulizia1205@gmail.com',
      color: '#ea4335'
    }
  ];

  const contactTypes = [
    {
      id: 'general',
      title: 'General Inquiry',
      icon: <MessageSquare />,
      description: 'Have a question or want to say hello?'
    },
    {
      id: 'collaboration',
      title: 'Collaboration',
      icon: <Users />,
      description: 'Interested in working together on a project?'
    },
    {
      id: 'opportunity',
      title: 'Job Opportunity',
      icon: <Briefcase />,
      description: 'Have a position that might be a good fit?'
    },
    {
      id: 'research',
      title: 'Research',
      icon: <Code2 />,
      description: 'Want to discuss research or academic work?'
    }
  ];

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleTypeChange = (type: FormData['type']) => {
    setFormData(prev => ({ ...prev, type }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    setFormStatus({ type: 'loading', message: 'Sending your message...' });
    
    // Simulate form submission
    setTimeout(() => {
      setFormStatus({
        type: 'success',
        message: 'Message sent successfully! I\'ll get back to you soon.'
      });
      
      // Reset form after success
      setTimeout(() => {
        setFormData({
          name: '',
          email: '',
          subject: '',
          message: '',
          type: 'general'
        });
        setFormStatus({ type: 'idle', message: '' });
      }, 5000);
    }, 2000);
  };

  return (
    <div className="contact-page">
      {/* Hero Section */}
      <section className="contact-hero">
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
            <Mail size={48} />
          </motion.div>
          <h1 className="hero-title">Let's Connect</h1>
          <p className="hero-subtitle">
            I'm always interested in new opportunities, collaborations, and interesting conversations
          </p>
          <div className="hero-badges">
            <motion.span 
              className="badge"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
            >
              <CheckCircle size={16} />
              {contactInfo.availability}
            </motion.span>
            <motion.span 
              className="badge"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.5 }}
            >
              <Clock size={16} />
              {contactInfo.responseTime}
            </motion.span>
          </div>
        </motion.div>
      </section>

      <div className="container">
        <div className="contact-content">
          {/* Contact Form */}
          <motion.div 
            className="contact-form-section"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h2>Send a Message</h2>
            
            {/* Contact Type Selection */}
            <div className="contact-types">
              {contactTypes.map((type) => (
                <motion.button
                  key={type.id}
                  className={`type-card ${formData.type === type.id ? 'active' : ''}`}
                  onClick={() => handleTypeChange(type.id as FormData['type'])}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {type.icon}
                  <div>
                    <h4>{type.title}</h4>
                    <p>{type.description}</p>
                  </div>
                </motion.button>
              ))}
            </div>

            <form onSubmit={handleSubmit} className="contact-form">
              <div className="form-row">
                <motion.div 
                  className={`form-group ${focusedField === 'name' ? 'focused' : ''}`}
                  whileHover={{ scale: 1.02 }}
                >
                  <label htmlFor="name">Your Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    onFocus={() => setFocusedField('name')}
                    onBlur={() => setFocusedField(null)}
                    required
                    placeholder="John Doe"
                  />
                </motion.div>
                
                <motion.div 
                  className={`form-group ${focusedField === 'email' ? 'focused' : ''}`}
                  whileHover={{ scale: 1.02 }}
                >
                  <label htmlFor="email">Email Address</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    onFocus={() => setFocusedField('email')}
                    onBlur={() => setFocusedField(null)}
                    required
                    placeholder="john@example.com"
                  />
                </motion.div>
              </div>
              
              <motion.div 
                className={`form-group ${focusedField === 'subject' ? 'focused' : ''}`}
                whileHover={{ scale: 1.02 }}
              >
                <label htmlFor="subject">Subject</label>
                <input
                  type="text"
                  id="subject"
                  name="subject"
                  value={formData.subject}
                  onChange={handleInputChange}
                  onFocus={() => setFocusedField('subject')}
                  onBlur={() => setFocusedField(null)}
                  required
                  placeholder="What's this about?"
                />
              </motion.div>
              
              <motion.div 
                className={`form-group ${focusedField === 'message' ? 'focused' : ''}`}
                whileHover={{ scale: 1.02 }}
              >
                <label htmlFor="message">Message</label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleInputChange}
                  onFocus={() => setFocusedField('message')}
                  onBlur={() => setFocusedField(null)}
                  required
                  rows={6}
                  placeholder="Tell me more about your project, idea, or question..."
                />
              </motion.div>
              
              <motion.button
                type="submit"
                className="submit-btn"
                disabled={formStatus.type === 'loading'}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {formStatus.type === 'loading' ? (
                  <>
                    <div className="spinner" />
                    <span>Sending...</span>
                  </>
                ) : (
                  <>
                    <Send size={20} />
                    <span>Send Message</span>
                  </>
                )}
              </motion.button>
              
              {formStatus.type !== 'idle' && (
                <motion.div 
                  className={`form-message ${formStatus.type}`}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  {formStatus.type === 'success' && <CheckCircle size={20} />}
                  {formStatus.type === 'error' && <AlertCircle size={20} />}
                  <span>{formStatus.message}</span>
                </motion.div>
              )}
            </form>
          </motion.div>

          {/* Contact Information */}
          <motion.div 
            className="contact-info-section"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="info-card">
              <h3>Contact Information</h3>
              
              <div className="info-items">
                <motion.div 
                  className="info-item"
                  whileHover={{ x: 5 }}
                >
                  <Mail className="info-icon" />
                  <div>
                    <label>Email</label>
                    <a href={`mailto:${contactInfo.email}`}>{contactInfo.email}</a>
                  </div>
                </motion.div>
                
                <motion.div 
                  className="info-item"
                  whileHover={{ x: 5 }}
                >
                  <Phone className="info-icon" />
                  <div>
                    <label>Phone</label>
                    <a href={`tel:${contactInfo.phone}`}>{contactInfo.phone}</a>
                  </div>
                </motion.div>
                
                <motion.div 
                  className="info-item"
                  whileHover={{ x: 5 }}
                >
                  <MapPin className="info-icon" />
                  <div>
                    <label>Location</label>
                    <span>{contactInfo.location}</span>
                  </div>
                </motion.div>
              </div>
            </div>

            <div className="info-card">
              <h3>Connect on Social</h3>
              
              <div className="social-links">
                {socialLinks.map((link, index) => (
                  <motion.a
                    key={link.name}
                    href={link.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="social-link"
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.1 * index }}
                    whileHover={{ scale: 1.1, y: -5 }}
                  >
                    <div className="social-icon" style={{ background: `${link.color}20`, color: link.color }}>
                      {link.icon}
                    </div>
                    <div className="social-info">
                      <span className="social-name">{link.name}</span>
                      <span className="social-username">{link.username}</span>
                    </div>
                  </motion.a>
                ))}
              </div>
            </div>

            <div className="info-card">
              <h3>Quick Actions</h3>
              
              <div className="quick-actions">
                <motion.a 
                  href="/resume.pdf"
                  download
                  className="action-link"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Download size={20} />
                  <span>Download Resume</span>
                  <ArrowRight size={16} />
                </motion.a>
                
                <motion.a 
                  href="https://calendly.com/jgulizia"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="action-link"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Calendar size={20} />
                  <span>Schedule a Meeting</span>
                  <ArrowRight size={16} />
                </motion.a>
                
                <motion.a 
                  href="#"
                  className="action-link"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Coffee size={20} />
                  <span>Buy Me a Coffee</span>
                  <ArrowRight size={16} />
                </motion.a>
              </div>
            </div>

            <div className="availability-card">
              <Sparkles className="availability-icon" />
              <h4>Current Status</h4>
              <p>Open to new opportunities and exciting projects. Let's create something amazing together!</p>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default ContactPage;