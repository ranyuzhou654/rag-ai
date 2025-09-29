'use client';

import { useState, useEffect } from 'react';
import { useAppStore } from '@/store/useAppStore';
import Sidebar from './Sidebar';
import Header from './Header';
import ChatInterface from '../chat/ChatInterface';
import SearchInterface from '../search/SearchInterface';
import TrendingInterface from '../trending/TrendingInterface';
import DashboardInterface from '../dashboard/DashboardInterface';
import { cn } from '@/lib/utils';

export default function MainLayout() {
  const { currentView, sidebarOpen, healthStatus } = useAppStore();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    );
  }

  const renderMainContent = () => {
    switch (currentView) {
      case 'chat':
        return <ChatInterface />;
      case 'search':
        return <SearchInterface />;
      case 'trending':
        return <TrendingInterface />;
      case 'dashboard':
        return <DashboardInterface />;
      default:
        return <ChatInterface />;
    }
  };

  return (
    <div className="h-screen flex bg-background text-foreground">
      {/* Sidebar */}
      <div
        className={cn(
          'transition-all duration-300 ease-in-out border-r border-border bg-card',
          sidebarOpen ? 'w-80' : 'w-0 overflow-hidden'
        )}
      >
        <Sidebar />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <Header />

        {/* Health Status Banner */}
        {healthStatus === 'critical' && (
          <div className="bg-destructive/20 border-b border-destructive/30 px-4 py-2">
            <div className="flex items-center gap-2 text-sm text-destructive">
              <div className="w-2 h-2 rounded-full bg-destructive animate-pulse" />
              系统状态异常，部分功能可能不可用
            </div>
          </div>
        )}

        {healthStatus === 'warning' && (
          <div className="bg-yellow-500/20 border-b border-yellow-500/30 px-4 py-2">
            <div className="flex items-center gap-2 text-sm text-yellow-700 dark:text-yellow-300">
              <div className="w-2 h-2 rounded-full bg-yellow-500" />
              系统性能可能受到影响
            </div>
          </div>
        )}

        {/* Main Content */}
        <main className="flex-1 overflow-hidden">
          {renderMainContent()}
        </main>
      </div>
    </div>
  );
}