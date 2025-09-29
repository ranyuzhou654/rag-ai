# RAG-AI Frontend

> **Modern Next.js Frontend for the RAG-AI Academic Research System**

A sophisticated React-based frontend interface built with Next.js 14, TypeScript, and Tailwind CSS for the RAG-AI academic paper research system. Features real-time chat, advanced search, citation management, and system monitoring.

## 🌟 Key Features

### 💬 **Intelligent Chat Interface**
- **Real-time streaming**: Server-Sent Events for live response generation
- **Session management**: Persistent chat history with Zustand state management
- **Rich message display**: Source citations, confidence scores, and timestamps
- **Responsive design**: Optimized for desktop, tablet, and mobile

### 🔍 **Advanced Search System**
- **Hybrid search modes**: Semantic, keyword, hybrid, and academic search
- **Smart filtering**: Author, year, source, and category filters
- **Batch operations**: Select and export multiple documents
- **Citation export**: APA, MLA, BibTeX, IEEE, and Chicago formats

### 📈 **System Monitoring Dashboard**
- **Real-time metrics**: Cache performance, system health, and usage statistics
- **Performance visualization**: Hit rates, response times, and error tracking
- **Health indicators**: Component status and system alerts
- **Auto-refresh**: Live data updates every 30 seconds

### 📚 **Trending Papers**
- **Academic trends**: Hot papers from ArXiv, HuggingFace, and blogs
- **Time-based filtering**: 7-day, 30-day, and 90-day trending analysis
- **Category filters**: AI, ML, NLP, Computer Vision, and more
- **Trending scores**: Algorithmic popularity and engagement metrics

## 🛠️ Technology Stack

### **Core Framework**
- **Next.js 14**: React framework with App Router and Server Components
- **TypeScript**: Full type safety throughout the application
- **React 18**: Latest React features with concurrent rendering

### **Styling & UI**
- **Tailwind CSS**: Utility-first CSS framework for rapid development
- **Custom Components**: Reusable UI components with consistent design
- **Responsive Design**: Mobile-first approach with breakpoint optimization
- **Dark/Light Mode**: System preference detection and manual toggle

### **State Management**
- **Zustand**: Lightweight state management with persistence
- **Session Storage**: Chat history and user preferences
- **Optimistic Updates**: Immediate UI feedback for better UX

### **API & Communication**
- **Axios**: HTTP client with request/response interceptors
- **Server-Sent Events**: Real-time streaming for chat responses
- **AsyncGenerator**: Modern streaming implementation
- **Error Handling**: Comprehensive error boundaries and retry logic

## 📁 Project Structure

```
frontend/
├── 📦 Configuration
│   ├── package.json              # Dependencies and scripts
│   ├── next.config.js           # Next.js configuration
│   ├── tailwind.config.ts       # Tailwind CSS configuration
│   ├── tsconfig.json           # TypeScript configuration
│   └── postcss.config.js       # PostCSS configuration
│
├── 🚀 Deployment
│   ├── Dockerfile              # Production container
│   ├── Dockerfile.dev          # Development container
│   └── .dockerignore          # Docker ignore rules
│
├── 📱 Application Source
│   └── src/
│       ├── 🎯 App Router
│       │   └── app/
│       │       ├── layout.tsx          # Root layout component
│       │       ├── page.tsx           # Home page
│       │       └── globals.css        # Global styles
│       │
│       ├── 🧩 Components
│       │   ├── layout/               # Layout components
│       │   │   ├── MainLayout.tsx    # Main application layout
│       │   │   ├── Header.tsx        # Navigation header
│       │   │   └── Sidebar.tsx       # Chat history sidebar
│       │   │
│       │   ├── chat/                 # Chat interface
│       │   │   └── ChatInterface.tsx # Real-time chat component
│       │   │
│       │   ├── search/              # Document search
│       │   │   └── SearchInterface.tsx # Advanced search interface
│       │   │
│       │   ├── trending/            # Trending papers
│       │   │   └── TrendingInterface.tsx # Hot papers display
│       │   │
│       │   ├── dashboard/           # System monitoring
│       │   │   └── DashboardInterface.tsx # Metrics dashboard
│       │   │
│       │   └── ui/                  # Reusable UI components
│       │       ├── Button.tsx       # Custom button component
│       │       └── Loading.tsx      # Loading indicators
│       │
│       ├── 🔧 Utilities & Services
│       │   └── lib/
│       │       ├── api.ts           # API client with streaming
│       │       └── utils.ts         # Helper functions
│       │
│       ├── 🏪 State Management
│       │   └── store/
│       │       └── useAppStore.ts   # Zustand global store
│       │
│       └── 📝 Type Definitions
│           └── types/
│               └── index.ts         # TypeScript interfaces
│
└── 📚 Documentation
    └── README.md                   # This file
```

## 🚀 Quick Start

### Development Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Set up environment variables
cp .env.example .env.local
# Edit .env.local with your API URLs

# 4. Start development server
npm run dev

# 5. Open browser
# http://localhost:3000
```

### Environment Variables

```bash
# .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NODE_ENV=development
```

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build production bundle
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript compiler
```

## 🎨 Key Components

### ChatInterface Component

The main chat interface with real-time streaming capabilities:

```tsx
// Real-time streaming implementation
const stream = apiClient.askQuestionStream({
  query: input.trim(),
  max_results: preferences.maxResults,
  include_sources: preferences.includeSources,
  rag_mode: preferences.defaultRAGMode,
  stream_response: true,
});

for await (const chunk of stream) {
  if (chunk.type === 'content' && chunk.content) {
    accumulatedContent += chunk.content;
    updateMessage(sessionId, messageId, {
      content: accumulatedContent,
    });
  }
}
```

### SearchInterface Component

Advanced document search with filtering and export:

```tsx
// Academic search with filters
const response = await apiClient.searchDocuments({
  query: searchQuery,
  search_type: 'academic',
  filters: {
    authors: selectedAuthors,
    year_range: [startYear, endYear],
    sources: selectedSources,
    has_full_text: true
  },
  limit: 50
});
```

### State Management with Zustand

Efficient state management with persistence:

```tsx
export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      // Chat state
      currentSession: null,
      sessions: [],
      
      // Actions
      createSession: () => {
        const newSession = {
          id: uuidv4(),
          title: '新对话',
          messages: [],
          created_at: new Date(),
        };
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSession: newSession,
        }));
        return newSession;
      },
    }),
    {
      name: 'rag-ai-store',
      partialize: (state) => ({
        sessions: state.sessions,
        preferences: state.preferences,
      }),
    }
  )
);
```

## 🎯 Features in Detail

### Real-Time Chat
- **Streaming responses**: Live text generation with Server-Sent Events
- **Message history**: Persistent chat sessions with local storage
- **Source citations**: Academic references with relevance scores
- **Feedback system**: Thumbs up/down for response quality

### Document Search
- **Multi-mode search**: Semantic, keyword, hybrid, and academic modes
- **Advanced filters**: Author, year, source type, and category filtering
- **Batch operations**: Multi-select with bulk export functionality
- **Citation management**: Export in multiple academic formats

### System Dashboard
- **Cache metrics**: Hit rates and performance across 4 cache layers
- **Health monitoring**: Component status and system alerts
- **Usage statistics**: Request patterns and user engagement
- **Real-time updates**: Live data refresh without page reload

### Trending Papers
- **Hot paper detection**: Algorithmic trending based on citations and engagement
- **Time-based analysis**: 7-day, 30-day, and 90-day trending periods
- **Category filtering**: Focus on specific AI/ML research areas
- **Popularity scoring**: Transparent trending algorithms

## 🎨 Design System

### Color Palette
- **Primary**: Blue gradient (#3B82F6 to #8B5CF6)
- **Secondary**: Neutral grays for text and backgrounds
- **Success**: Green (#10B981) for positive actions
- **Warning**: Yellow (#F59E0B) for alerts
- **Error**: Red (#EF4444) for errors

### Typography
- **Font**: Inter from Google Fonts
- **Headings**: Bold weights (600-700) for hierarchy
- **Body**: Regular (400) and medium (500) for readability
- **Code**: Monospace for technical content

### Components
- **Consistent spacing**: 4px grid system via Tailwind
- **Rounded corners**: 8px border radius for modern feel
- **Shadows**: Subtle elevation for depth
- **Animations**: Smooth transitions (200-300ms)

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 640px (sm)
- **Tablet**: 640px - 1024px (md-lg)  
- **Desktop**: > 1024px (xl+)

### Mobile Optimizations
- **Touch-friendly**: Larger tap targets (44px minimum)
- **Navigation**: Bottom tab bar for mobile devices
- **Typography**: Adjusted font sizes for readability
- **Performance**: Optimized images and lazy loading

## 🔧 Development Guidelines

### Code Style
- **TypeScript**: Strict type checking enabled
- **ESLint**: Consistent code formatting
- **Prettier**: Automatic code formatting
- **Components**: Functional components with hooks

### Performance
- **Lazy loading**: Dynamic imports for code splitting
- **Image optimization**: Next.js automatic image optimization
- **Bundle analysis**: Webpack bundle analyzer integration
- **Caching**: Aggressive caching strategies

### Testing (Planned)
- **Unit tests**: Jest and React Testing Library
- **Integration tests**: API endpoint testing
- **E2E tests**: Playwright for user flows
- **Visual regression**: Screenshot comparison testing

## 🚀 Deployment

### Production Build

```bash
# Build for production
npm run build

# Start production server
npm start
```

### Docker Deployment

```bash
# Build production image
docker build -t rag-ai-frontend .

# Run container
docker run -p 3000:3000 \
  -e NEXT_PUBLIC_API_URL=http://your-api-domain \
  rag-ai-frontend
```

### Environment Configuration

```bash
# Production environment variables
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com
NODE_ENV=production
```

## 🤝 Contributing

### Development Workflow

1. **Fork repository** and create feature branch
2. **Install dependencies** and start development server
3. **Implement features** following component patterns
4. **Add TypeScript types** for all new interfaces
5. **Test thoroughly** across different devices
6. **Submit pull request** with clear description

### Component Guidelines

```tsx
// Example component structure
interface ComponentProps {
  required: string;
  optional?: number;
  children?: React.ReactNode;
}

export default function Component({ 
  required, 
  optional = 0, 
  children 
}: ComponentProps) {
  // Hooks
  const [state, setState] = useState();
  
  // Event handlers
  const handleAction = () => {
    // Implementation
  };
  
  // Render
  return (
    <div className="component-styles">
      {children}
    </div>
  );
}
```

## 📈 Performance Metrics

### Bundle Size
- **First Load JS**: ~200KB (target < 250KB)
- **Code Splitting**: Automatic route-based splitting
- **Tree Shaking**: Unused code elimination

### Core Web Vitals
- **LCP**: < 2.5s (Largest Contentful Paint)
- **FID**: < 100ms (First Input Delay)  
- **CLS**: < 0.1 (Cumulative Layout Shift)

### Optimization Strategies
- **Image optimization**: WebP format with fallbacks
- **Font loading**: Preload critical fonts
- **CSS optimization**: Purge unused Tailwind classes
- **JavaScript**: ES modules with dynamic imports

## 📚 Resources

- **Next.js Documentation**: [nextjs.org/docs](https://nextjs.org/docs)
- **TypeScript Handbook**: [typescriptlang.org](https://www.typescriptlang.org/)
- **Tailwind CSS**: [tailwindcss.com](https://tailwindcss.com/)
- **Zustand Guide**: [github.com/pmndrs/zustand](https://github.com/pmndrs/zustand)

## 🎉 Acknowledgments

Built with modern frontend technologies:
- **Next.js 14** for the React framework
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **Zustand** for state management
- **Lucide React** for icons

---

*Frontend v2.0 - Modern interface for academic research and knowledge discovery* 🚀