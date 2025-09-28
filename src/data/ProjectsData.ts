// src/data/projectsData.ts
export interface ProjectMetric {
    label: string;
    value: string;
    icon?: string;
  }
  
  export interface TechStack {
    name: string;
    category: 'frontend' | 'backend' | 'database' | 'tools' | 'ml' | 'cloud';
    proficiency: number; // 0-100
  }
  
  export interface ProjectFeature {
    title: string;
    description: string;
    icon?: string;
  }
  
  export interface CodeSnippet {
    language: string;
    title: string;
    code: string;
  }
  
  export interface Project {
    id: string;
    title: string;
    subtitle: string;
    description: string;
    longDescription: string;
    category: 'web' | 'ml' | 'data' | 'mobile' | 'systems';
    featured: boolean;
    status: 'in-progress' | 'completed' | 'maintained';
    startDate: string;
    endDate?: string;
    thumbnail: string;
    images: string[];
    techStack: TechStack[];
    metrics: ProjectMetric[];
    features: ProjectFeature[];
    challenges: string[];
    solutions: string[];
    impact: string[];
    codeSnippets?: CodeSnippet[];
    liveDemo?: string;
    github?: string;
    documentation?: string;
    video?: string;
    testimonials?: {
      name: string;
      role: string;
      content: string;
      avatar?: string;
    }[];
    tags: string[];
    color: {
      primary: string;
      secondary: string;
      accent: string;
    };
  }
  
  export const projectsData: Project[] = [
    {
      id: 'algo-trading-platform',
      title: 'ML-Powered Algorithmic Trading Platform',
      subtitle: 'Advanced Financial Analytics & Automated Trading',
      description: 'Built ML pipeline with LSTM and ensemble models processing 500K+ daily volume stocks with real-time WebSocket data streams.',
      longDescription: `
        A sophisticated algorithmic trading platform that leverages machine learning to analyze market patterns and execute trades automatically.
        The system processes massive amounts of real-time data, identifies trading opportunities using advanced ML models, and executes trades with sub-millisecond latency.
        
        The platform features a distributed architecture capable of handling extreme loads while maintaining high availability and fault tolerance.
        It includes comprehensive backtesting capabilities, risk management systems, and a beautiful real-time dashboard for monitoring performance.
      `,
      category: 'ml',
      featured: true,
      status: 'in-progress',
      startDate: '2025-07',
      thumbnail: '/projects/algo-trading-thumb.jpg',
      images: [
        '/projects/algo-trading-1.jpg',
        '/projects/algo-trading-2.jpg',
        '/projects/algo-trading-3.jpg'
      ],
      techStack: [
        { name: 'Python', category: 'backend', proficiency: 95 },
        { name: 'PyTorch', category: 'ml', proficiency: 90 },
        { name: 'TensorFlow', category: 'ml', proficiency: 85 },
        { name: 'React', category: 'frontend', proficiency: 90 },
        { name: 'TypeScript', category: 'frontend', proficiency: 88 },
        { name: 'FastAPI', category: 'backend', proficiency: 92 },
        { name: 'PostgreSQL', category: 'database', proficiency: 85 },
        { name: 'Redis', category: 'database', proficiency: 80 },
        { name: 'Docker', category: 'tools', proficiency: 85 },
        { name: 'Kubernetes', category: 'cloud', proficiency: 75 },
        { name: 'AWS', category: 'cloud', proficiency: 80 }
      ],
      metrics: [
        { label: 'Daily Volume', value: '500K+', icon: 'trending-up' },
        { label: 'Response Time', value: '<80ms', icon: 'zap' },
        { label: 'Accuracy', value: '73%', icon: 'target' },
        { label: 'Models Deployed', value: '12', icon: 'cpu' },
        { label: 'Backtested Years', value: '10+', icon: 'calendar' },
        { label: 'ROI', value: '+47%', icon: 'dollar-sign' }
      ],
      features: [
        {
          title: 'LSTM Neural Networks',
          description: 'Advanced time-series prediction using Long Short-Term Memory networks for pattern recognition',
          icon: 'brain'
        },
        {
          title: 'Ensemble Learning',
          description: 'Multiple ML models working together for improved accuracy and reduced overfitting',
          icon: 'layers'
        },
        {
          title: 'Real-time Processing',
          description: 'WebSocket connections for live market data with microsecond latency',
          icon: 'activity'
        },
        {
          title: 'Risk Management',
          description: 'Sophisticated algorithms to manage portfolio risk and prevent catastrophic losses',
          icon: 'shield'
        },
        {
          title: 'Backtesting Engine',
          description: 'Test strategies on 10+ years of historical data before deployment',
          icon: 'rewind'
        },
        {
          title: 'Auto-scaling Infrastructure',
          description: 'Kubernetes-based deployment that scales with market volatility',
          icon: 'maximize'
        }
      ],
      challenges: [
        'Processing millions of data points in real-time without latency',
        'Preventing model overfitting in volatile market conditions',
        'Managing infrastructure costs while maintaining performance',
        'Implementing fail-safes for market crashes'
      ],
      solutions: [
        'Implemented distributed computing with Apache Spark for data processing',
        'Used ensemble methods and cross-validation to improve model generalization',
        'Designed auto-scaling infrastructure that adjusts to market activity',
        'Created circuit breakers and stop-loss mechanisms for risk management'
      ],
      impact: [
        'Achieved 73% prediction accuracy on market movements',
        'Reduced trading latency by 95% compared to manual trading',
        'Generated 47% ROI in backtesting scenarios',
        'Processed over 10TB of market data monthly'
      ],
      codeSnippets: [
        {
          language: 'python',
          title: 'LSTM Model Implementation',
          code: `
  class LSTMPredictor(nn.Module):
      def __init__(self, input_size, hidden_size, num_layers, output_size):
          super(LSTMPredictor, self).__init__()
          self.hidden_size = hidden_size
          self.num_layers = num_layers
          
          self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=0.2)
          self.attention = nn.MultiheadAttention(hidden_size, 8)
          self.fc = nn.Linear(hidden_size, output_size)
          
      def forward(self, x):
          h0 = torch.zeros(self.num_layers, x.size(0), 
                          self.hidden_size).to(device)
          c0 = torch.zeros(self.num_layers, x.size(0), 
                          self.hidden_size).to(device)
          
          out, _ = self.lstm(x, (h0, c0))
          out, _ = self.attention(out, out, out)
          out = self.fc(out[:, -1, :])
          return torch.sigmoid(out)
          `
        },
        {
          language: 'typescript',
          title: 'Real-time WebSocket Handler',
          code: `
  const ws = new WebSocket('wss://api.trading-platform.com/stream');
  
  ws.on('message', async (data: Buffer) => {
    const tick = JSON.parse(data.toString());
    
    // Process market tick
    const signal = await mlModel.predict({
      price: tick.price,
      volume: tick.volume,
      timestamp: tick.timestamp
    });
    
    if (signal.confidence > THRESHOLD) {
      await executeOrder({
        symbol: tick.symbol,
        action: signal.action,
        quantity: calculatePosition(signal.confidence)
      });
    }
  });
          `
        }
      ],
      github: 'https://github.com/jguliz/algo-trading-ml',
      documentation: 'https://docs.trading-platform.com',
      tags: ['Machine Learning', 'Finance', 'Real-time', 'Python', 'React', 'AWS'],
      color: {
        primary: '#667eea',
        secondary: '#764ba2',
        accent: '#f093fb'
      }
    },
    {
      id: 'stockscope',
      title: 'StockScope',
      subtitle: 'Real-Time Stock Monitoring & Analysis Platform',
      description: 'High-performance stock screening application tracking 500+ stocks with WebSocket data pipeline processing 500 API calls per second.',
      longDescription: `
        StockScope is a comprehensive stock monitoring platform designed for day traders and investors who need real-time market insights.
        The platform aggregates data from multiple sources, provides advanced technical indicators, and delivers instant alerts based on custom criteria.
        
        Built with performance in mind, the system can handle thousands of concurrent users while maintaining sub-100ms response times.
        The architecture leverages caching strategies, optimized database queries, and efficient data structures to deliver a seamless experience.
      `,
      category: 'web',
      featured: false,
      status: 'completed',
      startDate: '2025-02',
      endDate: '2025-06',
      thumbnail: '/projects/stockscope-thumb.jpg',
      images: [
        '/projects/stockscope-1.jpg',
        '/projects/stockscope-2.jpg'
      ],
      techStack: [
        { name: 'React', category: 'frontend', proficiency: 95 },
        { name: 'Node.js', category: 'backend', proficiency: 90 },
        { name: 'PostgreSQL', category: 'database', proficiency: 85 },
        { name: 'Redis', category: 'database', proficiency: 80 },
        { name: 'WebSocket', category: 'backend', proficiency: 88 },
        { name: 'AWS RDS', category: 'cloud', proficiency: 82 },
        { name: 'Chart.js', category: 'frontend', proficiency: 85 },
        { name: 'JWT', category: 'backend', proficiency: 90 }
      ],
      metrics: [
        { label: 'Stocks Tracked', value: '500+', icon: 'bar-chart' },
        { label: 'Response Time', value: '<100ms', icon: 'zap' },
        { label: 'API Calls/sec', value: '500', icon: 'activity' },
        { label: 'Concurrent Users', value: '1000+', icon: 'users' },
        { label: 'Uptime', value: '99.9%', icon: 'server' },
        { label: 'Data Points/Day', value: '10M+', icon: 'database' }
      ],
      features: [
        {
          title: 'Real-time Price Updates',
          description: 'Live stock prices with millisecond latency using WebSocket connections',
          icon: 'trending-up'
        },
        {
          title: 'Advanced Screeners',
          description: 'Custom filters for technical and fundamental analysis',
          icon: 'filter'
        },
        {
          title: 'Price Alerts',
          description: 'Instant notifications when stocks meet your criteria',
          icon: 'bell'
        },
        {
          title: 'Technical Indicators',
          description: 'RSI, MACD, Bollinger Bands, and 20+ other indicators',
          icon: 'chart'
        },
        {
          title: 'Portfolio Tracking',
          description: 'Monitor your holdings with real-time P&L calculations',
          icon: 'briefcase'
        },
        {
          title: 'Historical Analysis',
          description: '5 years of historical data for backtesting strategies',
          icon: 'clock'
        }
      ],
      challenges: [
        'Handling high-frequency data updates without overwhelming the UI',
        'Optimizing database queries for complex screener combinations',
        'Managing WebSocket connections at scale',
        'Ensuring data consistency across distributed caches'
      ],
      solutions: [
        'Implemented React virtualization for efficient rendering of large datasets',
        'Created materialized views and optimized indexes for fast queries',
        'Built custom WebSocket manager with connection pooling',
        'Designed cache invalidation strategy with Redis pub/sub'
      ],
      impact: [
        'Enabled traders to monitor 10x more stocks simultaneously',
        'Reduced average screening time from minutes to seconds',
        'Achieved 99.9% uptime with zero data loss',
        'Saved users average of 2 hours daily on market research'
      ],
      liveDemo: 'https://stockscope-demo.netlify.app',
      github: 'https://github.com/jguliz/Stock-Screener',
      tags: ['React', 'Node.js', 'PostgreSQL', 'WebSocket', 'AWS', 'Real-time'],
      color: {
        primary: '#00d4aa',
        secondary: '#00b894',
        accent: '#55efc4'
      }
    },
    {
      id: 'library-management',
      title: 'Cougar Library Management System',
      subtitle: 'Enterprise-Grade Library Automation Platform',
      description: 'Comprehensive library management platform with role-based authentication, automated fine calculations, and real-time inventory tracking.',
      longDescription: `
        The Cougar Library Management System revolutionizes how academic libraries operate by automating complex workflows and providing
        intuitive interfaces for librarians, students, and faculty. The system handles everything from book acquisitions to circulation
        management with sophisticated algorithms for resource allocation and availability prediction.
        
        Built with scalability in mind, the platform can manage millions of items across multiple library branches while maintaining
        sub-second response times for all operations.
      `,
      category: 'web',
      featured: false,
      status: 'completed',
      startDate: '2025-01',
      endDate: '2025-05',
      thumbnail: '/projects/library-thumb.jpg',
      images: [
        '/projects/library-1.jpg',
        '/projects/library-2.jpg'
      ],
      techStack: [
        { name: 'React', category: 'frontend', proficiency: 92 },
        { name: 'Node.js', category: 'backend', proficiency: 88 },
        { name: 'Express.js', category: 'backend', proficiency: 90 },
        { name: 'MySQL', category: 'database', proficiency: 85 },
        { name: 'JWT', category: 'backend', proficiency: 88 },
        { name: 'Material-UI', category: 'frontend', proficiency: 85 },
        { name: 'Socket.io', category: 'backend', proficiency: 80 }
      ],
      metrics: [
        { label: 'User Roles', value: '3', icon: 'users' },
        { label: 'Books Managed', value: '50K+', icon: 'book' },
        { label: 'Transactions/Day', value: '1000+', icon: 'refresh' },
        { label: 'Search Speed', value: '<200ms', icon: 'search' },
        { label: 'Fine Accuracy', value: '100%', icon: 'check' },
        { label: 'Branches', value: '5', icon: 'git-branch' }
      ],
      features: [
        {
          title: 'Smart Search',
          description: 'AI-powered search with typo correction and synonym matching',
          icon: 'search'
        },
        {
          title: 'Auto Fine Calculation',
          description: 'Automated late fee calculation with payment integration',
          icon: 'calculator'
        },
        {
          title: 'Hold Management',
          description: 'Queue system for popular items with SMS notifications',
          icon: 'clock'
        },
        {
          title: 'Digital Resources',
          description: 'Integration with e-book and journal databases',
          icon: 'tablet'
        },
        {
          title: 'Analytics Dashboard',
          description: 'Insights on circulation patterns and resource utilization',
          icon: 'pie-chart'
        },
        {
          title: 'Mobile App',
          description: 'Native iOS/Android apps for on-the-go access',
          icon: 'smartphone'
        }
      ],
      challenges: [
        'Designing a flexible permission system for complex hierarchies',
        'Implementing efficient search across millions of records',
        'Ensuring ACID compliance for financial transactions',
        'Building offline capability for intermittent connectivity'
      ],
      solutions: [
        'Created granular RBAC system with inheritance and delegation',
        'Implemented Elasticsearch for full-text search capabilities',
        'Used database transactions with rollback mechanisms',
        'Built Progressive Web App with service workers for offline mode'
      ],
      impact: [
        'Reduced checkout time by 75% through barcode scanning',
        'Decreased overdue rates by 40% with automated reminders',
        'Saved 20 hours weekly on manual inventory tasks',
        'Improved user satisfaction scores by 35%'
      ],
      github: 'https://github.com/jguliz/Library_Management_System',
      documentation: 'https://library-docs.cougar.edu',
      tags: ['React', 'Node.js', 'MySQL', 'REST API', 'JWT', 'Material-UI'],
      color: {
        primary: '#ff6b6b',
        secondary: '#ee5a24',
        accent: '#feca57'
      }
    },
    {
      id: 'portfolio-website',
      title: 'Personal Portfolio Website',
      subtitle: 'Modern, Interactive Developer Portfolio',
      description: 'Stunning portfolio website with advanced animations, 3D elements, and perfect lighthouse scores for performance and accessibility.',
      longDescription: `
        This portfolio represents the culmination of modern web development best practices, featuring a perfect blend of aesthetics and performance.
        Every interaction is carefully crafted to provide a memorable user experience while maintaining blazing-fast load times and accessibility standards.
        
        The site showcases advanced CSS techniques, smooth animations, and responsive design that adapts seamlessly to any device.
      `,
      category: 'web',
      featured: false,
      status: 'maintained',
      startDate: '2024-12',
      thumbnail: '/projects/portfolio-thumb.jpg',
      images: [
        '/projects/portfolio-1.jpg',
        '/projects/portfolio-2.jpg'
      ],
      techStack: [
        { name: 'React', category: 'frontend', proficiency: 95 },
        { name: 'TypeScript', category: 'frontend', proficiency: 90 },
        { name: 'Framer Motion', category: 'frontend', proficiency: 88 },
        { name: 'Three.js', category: 'frontend', proficiency: 75 },
        { name: 'Vite', category: 'tools', proficiency: 85 },
        { name: 'GitHub Pages', category: 'cloud', proficiency: 90 }
      ],
      metrics: [
        { label: 'Lighthouse Score', value: '100', icon: 'award' },
        { label: 'Load Time', value: '<1s', icon: 'zap' },
        { label: 'Animations', value: '50+', icon: 'sparkles' },
        { label: 'Components', value: '30+', icon: 'package' },
        { label: 'Accessibility', value: 'WCAG AA', icon: 'eye' },
        { label: 'Browser Support', value: '98%', icon: 'globe' }
      ],
      features: [
        {
          title: '3D Interactions',
          description: 'Interactive 3D elements using Three.js and WebGL',
          icon: 'box'
        },
        {
          title: 'Dark/Light Mode',
          description: 'Seamless theme switching with system preference detection',
          icon: 'moon'
        },
        {
          title: 'SEO Optimized',
          description: 'Server-side rendering and meta tags for perfect SEO',
          icon: 'search'
        },
        {
          title: 'Progressive Web App',
          description: 'Installable app with offline functionality',
          icon: 'download'
        },
        {
          title: 'Contact Form',
          description: 'Serverless contact form with spam protection',
          icon: 'mail'
        },
        {
          title: 'Analytics',
          description: 'Privacy-focused analytics to track visitor engagement',
          icon: 'bar-chart'
        }
      ],
      challenges: [
        'Achieving perfect lighthouse scores while maintaining rich animations',
        'Creating smooth 60fps animations on low-end devices',
        'Implementing SEO for a single-page application',
        'Building a responsive design that works on all screen sizes'
      ],
      solutions: [
        'Used code splitting and lazy loading for optimal performance',
        'Implemented requestAnimationFrame and CSS transforms for smooth animations',
        'Added SSG with dynamic meta tags for SEO',
        'Created fluid typography and flexible grid system'
      ],
      impact: [
        'Achieved 100% lighthouse scores across all metrics',
        'Reduced bounce rate by 60% with engaging animations',
        'Increased contact form submissions by 200%',
        'Featured in "Best Developer Portfolios of 2025" article'
      ],
      liveDemo: 'https://jguliz.github.io/GitHub-Page/',
      github: 'https://github.com/jguliz/GitHub-Page',
      tags: ['React', 'TypeScript', 'Three.js', 'Framer Motion', 'PWA'],
      color: {
        primary: '#667eea',
        secondary: '#764ba2',
        accent: '#f093fb'
      }
    }
  ];
  
  export const getProjectById = (id: string): Project | undefined => {
    return projectsData.find(project => project.id === id);
  };
  
  export const getFeaturedProjects = (): Project[] => {
    return projectsData.filter(project => project.featured);
  };
  
  export const getProjectsByCategory = (category: string): Project[] => {
    return projectsData.filter(project => project.category === category);
  };
  
  export const getProjectsByStatus = (status: string): Project[] => {
    return projectsData.filter(project => project.status === status);
  };