// Global app state management with Zustand
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { v4 as uuidv4 } from 'uuid';
import {
  AppState,
  ChatSession,
  Message,
  Document,
  TrendingPaper,
  SystemStats,
  UserPreferences,
} from '@/types';

interface AppStore extends AppState {
  // Chat actions
  createSession: () => ChatSession;
  setCurrentSession: (session: ChatSession | null) => void;
  addMessage: (sessionId: string, message: Omit<Message, 'id' | 'timestamp'>) => void;
  updateMessage: (sessionId: string, messageId: string, updates: Partial<Message>) => void;
  deleteSession: (sessionId: string) => void;
  clearAllSessions: () => void;
  setIsTyping: (typing: boolean) => void;

  // Search actions
  setSearchResults: (results: Document[]) => void;
  setSearchLoading: (loading: boolean) => void;
  setTrendingPapers: (papers: TrendingPaper[]) => void;

  // System actions
  setSystemStats: (stats: SystemStats) => void;
  setHealthStatus: (status: 'healthy' | 'warning' | 'critical' | 'unknown') => void;

  // User actions
  updatePreferences: (preferences: Partial<UserPreferences>) => void;

  // UI actions
  setSidebarOpen: (open: boolean) => void;
  setCurrentView: (view: 'chat' | 'search' | 'trending' | 'dashboard') => void;
  toggleSidebar: () => void;
}

const defaultPreferences: UserPreferences = {
  theme: 'system',
  defaultRAGMode: 'ultimate',
  maxResults: 5,
  includeSources: true,
  citationFormat: 'apa',
  language: 'en',
};

export const useAppStore = create<AppStore>()(
  persist(
    (set, get) => ({
      // Initial state
      currentSession: null,
      sessions: [],
      isTyping: false,
      searchResults: [],
      searchLoading: false,
      trendingPapers: [],
      systemStats: null,
      healthStatus: 'unknown',
      preferences: defaultPreferences,
      sidebarOpen: true,
      currentView: 'chat',

      // Chat actions
      createSession: () => {
        const newSession: ChatSession = {
          id: uuidv4(),
          title: '新对话',
          messages: [],
          created_at: new Date(),
          updated_at: new Date(),
        };

        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSession: newSession,
        }));

        return newSession;
      },

      setCurrentSession: (session) => {
        set({ currentSession: session });
      },

      addMessage: (sessionId, messageData) => {
        const message: Message = {
          ...messageData,
          id: uuidv4(),
          timestamp: new Date(),
        };

        set((state) => {
          const sessions = state.sessions.map((session) => {
            if (session.id === sessionId) {
              const updatedSession = {
                ...session,
                messages: [...session.messages, message],
                updated_at: new Date(),
              };

              // Update title if this is the first user message
              if (
                message.role === 'user' &&
                session.messages.length === 0 &&
                session.title === '新对话'
              ) {
                updatedSession.title = message.content.slice(0, 50) + '...';
              }

              return updatedSession;
            }
            return session;
          });

          const currentSession = sessions.find((s) => s.id === sessionId) || null;

          return {
            sessions,
            currentSession,
          };
        });
      },

      updateMessage: (sessionId, messageId, updates) => {
        set((state) => {
          const sessions = state.sessions.map((session) => {
            if (session.id === sessionId) {
              return {
                ...session,
                messages: session.messages.map((message) =>
                  message.id === messageId ? { ...message, ...updates } : message
                ),
                updated_at: new Date(),
              };
            }
            return session;
          });

          const currentSession = sessions.find((s) => s.id === sessionId) || null;

          return {
            sessions,
            currentSession,
          };
        });
      },

      deleteSession: (sessionId) => {
        set((state) => {
          const sessions = state.sessions.filter((s) => s.id !== sessionId);
          const currentSession =
            state.currentSession?.id === sessionId ? null : state.currentSession;

          return {
            sessions,
            currentSession,
          };
        });
      },

      clearAllSessions: () => {
        set({
          sessions: [],
          currentSession: null,
        });
      },

      setIsTyping: (typing) => {
        set({ isTyping: typing });
      },

      // Search actions
      setSearchResults: (results) => {
        set({ searchResults: results });
      },

      setSearchLoading: (loading) => {
        set({ searchLoading: loading });
      },

      setTrendingPapers: (papers) => {
        set({ trendingPapers: papers });
      },

      // System actions
      setSystemStats: (stats) => {
        set({ systemStats: stats });
      },

      setHealthStatus: (status) => {
        set({ healthStatus: status });
      },

      // User actions
      updatePreferences: (newPreferences) => {
        set((state) => ({
          preferences: { ...state.preferences, ...newPreferences },
        }));
      },

      // UI actions
      setSidebarOpen: (open) => {
        set({ sidebarOpen: open });
      },

      setCurrentView: (view) => {
        set({ currentView: view });
      },

      toggleSidebar: () => {
        set((state) => ({ sidebarOpen: !state.sidebarOpen }));
      },
    }),
    {
      name: 'rag-ai-store',
      partialize: (state) => ({
        sessions: state.sessions,
        preferences: state.preferences,
        sidebarOpen: state.sidebarOpen,
        currentView: state.currentView,
      }),
    }
  )
);

// Selectors for commonly used state
export const useCurrentSession = () => useAppStore((state) => state.currentSession);
export const useSessions = () => useAppStore((state) => state.sessions);
export const usePreferences = () => useAppStore((state) => state.preferences);
export const useSearchResults = () => useAppStore((state) => state.searchResults);
export const useTrendingPapers = () => useAppStore((state) => state.trendingPapers);
export const useSystemStats = () => useAppStore((state) => state.systemStats);
export const useHealthStatus = () => useAppStore((state) => state.healthStatus);
export const useIsTyping = () => useAppStore((state) => state.isTyping);
export const useSidebarOpen = () => useAppStore((state) => state.sidebarOpen);
export const useCurrentView = () => useAppStore((state) => state.currentView);