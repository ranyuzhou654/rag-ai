'use client';

import { Menu, Search, Settings, BarChart3, TrendingUp, MessageSquare } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { Button } from '@/components/ui/Button';
import { cn } from '@/lib/utils';

export default function Header() {
  const { 
    currentView, 
    setCurrentView, 
    toggleSidebar, 
    sidebarOpen,
    healthStatus 
  } = useAppStore();

  const navigationItems = [
    { id: 'chat', label: '智能对话', icon: MessageSquare },
    { id: 'search', label: '文档搜索', icon: Search },
    { id: 'trending', label: '热门论文', icon: TrendingUp },
    { id: 'dashboard', label: '系统监控', icon: BarChart3 },
  ];

  const getHealthStatusColor = () => {
    switch (healthStatus) {
      case 'healthy':
        return 'bg-green-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'critical':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm px-4 flex items-center justify-between">
      {/* Left Section */}
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="p-2"
        >
          <Menu className="h-5 w-5" />
        </Button>

        <div className="flex items-center gap-2">
          <div className="font-bold text-xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            RAG-AI
          </div>
          <div className="text-sm text-muted-foreground">
            智能检索增强生成系统
          </div>
        </div>
      </div>

      {/* Center Navigation */}
      <nav className="hidden md:flex items-center gap-1 bg-muted/50 rounded-lg p-1">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          return (
            <Button
              key={item.id}
              variant={currentView === item.id ? "default" : "ghost"}
              size="sm"
              onClick={() => setCurrentView(item.id as any)}
              className={cn(
                "gap-2 px-3 py-2 text-sm font-medium transition-all",
                currentView === item.id && "shadow-sm"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Button>
          );
        })}
      </nav>

      {/* Right Section */}
      <div className="flex items-center gap-3">
        {/* System Health Indicator */}
        <div className="flex items-center gap-2">
          <div className={cn(
            "w-2 h-2 rounded-full",
            getHealthStatusColor(),
            healthStatus === 'healthy' && "animate-pulse"
          )} />
          <span className="text-sm text-muted-foreground hidden sm:inline">
            {healthStatus === 'healthy' && '系统正常'}
            {healthStatus === 'warning' && '系统警告'}
            {healthStatus === 'critical' && '系统故障'}
            {healthStatus === 'unknown' && '状态未知'}
          </span>
        </div>

        {/* Settings Button */}
        <Button
          variant="ghost"
          size="sm"
          className="p-2"
        >
          <Settings className="h-4 w-4" />
        </Button>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden fixed bottom-0 left-0 right-0 bg-card border-t border-border p-2 z-50">
        <div className="flex justify-around">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <Button
                key={item.id}
                variant={currentView === item.id ? "default" : "ghost"}
                size="sm"
                onClick={() => setCurrentView(item.id as any)}
                className="flex-1 gap-2 py-3 text-xs"
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Button>
            );
          })}
        </div>
      </div>
    </header>
  );
}