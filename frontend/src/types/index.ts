// Types for RAG-AI Frontend

export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  sources?: Source[];
  confidence?: number;
  cached?: boolean;
}

export interface Source {
  id: string;
  title: string;
  authors: string[];
  citation: string;
  links: {
    [key: string]: string;
  };
  relevance_score: number;
  abstract?: string;
  published_date?: string;
  source_type: 'arxiv' | 'huggingface' | 'blog' | 'journal';
}

export interface QueryRequest {
  query: string;
  max_results?: number;
  include_sources?: boolean;
  stream_response?: boolean;
  rag_mode?: 'basic' | 'enhanced' | 'agentic' | 'ultimate';
  filters?: SearchFilters;
}

export interface QueryResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  processing_time: number;
  query_id: string;
  cached: boolean;
}

export interface SearchFilters {
  authors?: string[];
  year_range?: [number, number];
  sources?: string[];
  categories?: string[];
  has_full_text?: boolean;
  language?: string;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  offset?: number;
  filters?: SearchFilters;
  search_type?: 'semantic' | 'keyword' | 'hybrid' | 'academic';
}

export interface Document {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published_date: string;
  source: string;
  url: string;
  categories: string[];
  full_text?: string;
  citation?: string;
  links?: { [key: string]: string };
}

export interface SystemStats {
  cache: {
    overall: {
      total_requests: number;
      total_hits: number;
      hit_rate: number;
      layer_hits: { [key: string]: number };
    };
    layers: {
      memory: CacheLayerStats;
      redis: CacheLayerStats;
      file: CacheLayerStats;
      vector: CacheLayerStats;
    };
  };
  citations: {
    total_sources: number;
    total_citations: number;
    source_types: { [key: string]: number };
    recent_activity: { [key: string]: number };
    average_citations_per_source: number;
  };
  feedback: {
    total_feedback: number;
    average_rating: number;
    recent_feedback: any[];
  };
  timestamp: string;
}

export interface CacheLayerStats {
  type: string;
  size: number;
  hits: number;
  misses: number;
  hit_rate: number;
  connected?: boolean;
  total_size_bytes?: number;
  memory_usage?: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  created_at: Date;
  updated_at: Date;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  defaultRAGMode: string;
  maxResults: number;
  includeSources: boolean;
  citationFormat: 'apa' | 'mla' | 'bibtex' | 'ieee' | 'chicago';
  language: 'en' | 'zh';
}

export interface FeedbackRequest {
  query_id: string;
  rating: number;
  comment?: string;
  user_id?: string;
}

export interface TrendingPaper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published_date: string;
  source: string;
  url: string;
  categories: string[];
  trending_score?: number;
}

export interface StreamChunk {
  type: 'start' | 'content' | 'sources' | 'complete' | 'error';
  content?: string;
  sources?: Source[];
  query_id?: string;
  error?: string;
  metadata?: any;
}

// API Response types
export interface ApiResponse<T = any> {
  data?: T;
  error?: string;
  status: 'success' | 'error';
  timestamp: string;
}

// Component Props
export interface BaseComponentProps {
  className?: string;
  children?: React.ReactNode;
}

export interface LoadingProps extends BaseComponentProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

// Store types
export interface AppState {
  // Chat state
  currentSession: ChatSession | null;
  sessions: ChatSession[];
  isTyping: boolean;
  
  // Search state
  searchResults: Document[];
  searchLoading: boolean;
  trendingPapers: TrendingPaper[];
  
  // System state
  systemStats: SystemStats | null;
  healthStatus: 'healthy' | 'warning' | 'critical' | 'unknown';
  
  // User state
  preferences: UserPreferences;
  
  // UI state
  sidebarOpen: boolean;
  currentView: 'chat' | 'search' | 'trending' | 'dashboard';
}