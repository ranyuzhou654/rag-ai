'use client';

import { useState } from 'react';
import { Plus, MessageSquare, Clock, Search, Trash2, Settings, Download } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { Button } from '@/components/ui/Button';
import { formatDistanceToNow } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import { cn } from '@/lib/utils';

export default function Sidebar() {
  const {
    sessions,
    currentSession,
    createSession,
    setCurrentSession,
    deleteSession,
    clearAllSessions,
  } = useAppStore();

  const [searchTerm, setSearchTerm] = useState('');

  const filteredSessions = sessions.filter(session =>
    session.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleNewChat = () => {
    const newSession = createSession();
    setCurrentSession(newSession);
  };

  const handleDeleteSession = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteSession(sessionId);
  };

  return (
    <div className="h-full flex flex-col bg-card">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <Button
          onClick={handleNewChat}
          className="w-full gap-2 bg-primary hover:bg-primary/90"
          size="sm"
        >
          <Plus className="h-4 w-4" />
          新建对话
        </Button>
      </div>

      {/* Search */}
      <div className="p-4 border-b border-border">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="搜索对话历史..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm bg-background border border-border rounded-md focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
          />
        </div>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto">
        {filteredSessions.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground">
            <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">暂无对话记录</p>
          </div>
        ) : (
          <div className="p-2 space-y-1">
            {filteredSessions.map((session) => (
              <div
                key={session.id}
                onClick={() => setCurrentSession(session)}
                className={cn(
                  "group relative p-3 rounded-lg cursor-pointer transition-all duration-200 hover:bg-accent/50",
                  currentSession?.id === session.id && "bg-accent shadow-sm border border-border"
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-sm truncate mb-1">
                      {session.title}
                    </h3>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>
                        {formatDistanceToNow(session.updated_at, {
                          addSuffix: true,
                          locale: zhCN,
                        })}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {session.messages.length} 条消息
                    </div>
                  </div>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => handleDeleteSession(session.id, e)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 h-auto w-auto text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer Actions */}
      <div className="p-4 border-t border-border space-y-2">
        {sessions.length > 0 && (
          <Button
            variant="outline"
            size="sm"
            onClick={clearAllSessions}
            className="w-full gap-2 text-xs"
          >
            <Trash2 className="h-3 w-3" />
            清空所有对话
          </Button>
        )}
        
        <Button
          variant="outline"
          size="sm"
          className="w-full gap-2 text-xs"
        >
          <Download className="h-3 w-3" />
          导出对话记录
        </Button>

        <Button
          variant="outline"
          size="sm"
          className="w-full gap-2 text-xs"
        >
          <Settings className="h-3 w-3" />
          设置偏好
        </Button>
      </div>
    </div>
  );
}