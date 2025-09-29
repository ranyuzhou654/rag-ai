'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Square, FileText, ExternalLink, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import { Message, Source } from '@/types';
import { cn, copyToClipboard } from '@/lib/utils';
import { formatDistanceToNow } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import Loading from '@/components/ui/Loading';

export default function ChatInterface() {
  const {
    currentSession,
    createSession,
    setCurrentSession,
    addMessage,
    updateMessage,
    isTyping,
    setIsTyping,
    preferences,
  } = useAppStore();

  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!currentSession) {
      const newSession = createSession();
      setCurrentSession(newSession);
    }
  }, [currentSession, createSession, setCurrentSession]);

  useEffect(() => {
    scrollToBottom();
  }, [currentSession?.messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!input.trim() || !currentSession || isStreaming) return;

    const userMessage: Omit<Message, 'id' | 'timestamp'> = {
      content: input.trim(),
      role: 'user',
    };

    addMessage(currentSession.id, userMessage);
    setInput('');
    setIsStreaming(true);
    setIsTyping(true);

    try {
      const assistantMessageId = `temp-${Date.now()}`;
      const assistantMessage: Omit<Message, 'id' | 'timestamp'> = {
        content: '',
        role: 'assistant',
        sources: [],
      };

      addMessage(currentSession.id, assistantMessage);

      const stream = apiClient.askQuestionStream({
        query: input.trim(),
        max_results: preferences.maxResults,
        include_sources: preferences.includeSources,
        rag_mode: preferences.defaultRAGMode as any,
        stream_response: true,
      });

      let accumulatedContent = '';
      let sources: Source[] = [];

      for await (const chunk of stream) {
        if (chunk.type === 'content' && chunk.content) {
          accumulatedContent += chunk.content;
          updateMessage(currentSession.id, assistantMessageId, {
            content: accumulatedContent,
          });
        } else if (chunk.type === 'sources' && chunk.sources) {
          sources = chunk.sources;
          updateMessage(currentSession.id, assistantMessageId, {
            sources: sources,
          });
        } else if (chunk.type === 'error') {
          updateMessage(currentSession.id, assistantMessageId, {
            content: `错误: ${chunk.error}`,
          });
          break;
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
      addMessage(currentSession.id, {
        content: '抱歉，处理您的请求时出现了错误。请稍后再试。',
        role: 'assistant',
      });
    } finally {
      setIsStreaming(false);
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopyMessage = async (content: string) => {
    try {
      await copyToClipboard(content);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
  };

  useEffect(() => {
    adjustTextareaHeight();
  }, [input]);

  if (!currentSession) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loading text="初始化对话..." />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        {currentSession.messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto">
            <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
              <FileText className="w-8 h-8 text-primary" />
            </div>
            <h2 className="text-xl font-semibold mb-2">开始新的对话</h2>
            <p className="text-muted-foreground mb-6">
              我是您的智能RAG助手，可以帮您搜索和分析学术文档，回答技术问题。
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full">
              {[
                '解释什么是Transformer架构',
                '搜索最新的LLM研究进展',
                '比较不同的向量数据库',
                '分析RAG系统的优化方法',
              ].map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => setInput(suggestion)}
                  className="p-3 text-left border border-border rounded-lg hover:bg-accent transition-colors text-sm"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          currentSession.messages.map((message, index) => (
            <MessageBubble
              key={message.id || index}
              message={message}
              onCopy={handleCopyMessage}
            />
          ))
        )}

        {isTyping && (
          <div className="flex items-start gap-3">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-sm font-medium">
              AI
            </div>
            <div className="flex-1 bg-muted rounded-lg p-4">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card/50 backdrop-blur-sm p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="relative flex items-end gap-2">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="输入您的问题..."
                disabled={isStreaming}
                className="w-full resize-none rounded-lg border border-border bg-background px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ minHeight: '44px', maxHeight: '120px' }}
                rows={1}
              />
            </div>
            
            <Button
              type="submit"
              disabled={!input.trim() || isStreaming}
              size="lg"
              className="h-11 px-4"
            >
              {isStreaming ? (
                <Square className="h-4 w-4" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          
          <div className="mt-2 text-xs text-muted-foreground">
            按 Enter 发送，Shift + Enter 换行
          </div>
        </form>
      </div>
    </div>
  );
}

function MessageBubble({ 
  message, 
  onCopy 
}: { 
  message: Message; 
  onCopy: (content: string) => void; 
}) {
  const isUser = message.role === 'user';
  
  return (
    <div className={cn(
      'flex items-start gap-3',
      isUser && 'flex-row-reverse'
    )}>
      {/* Avatar */}
      <div className={cn(
        'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium',
        isUser ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'
      )}>
        {isUser ? '您' : 'AI'}
      </div>

      {/* Message Content */}
      <div className={cn(
        'flex-1 max-w-[80%]',
        isUser && 'flex flex-col items-end'
      )}>
        <div className={cn(
          'rounded-lg px-4 py-3 text-sm',
          isUser 
            ? 'bg-primary text-primary-foreground ml-auto' 
            : 'bg-muted'
        )}>
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
          
          {message.confidence && (
            <div className="mt-2 text-xs opacity-70">
              置信度: {(message.confidence * 100).toFixed(1)}%
            </div>
          )}
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 space-y-2 max-w-full">
            <div className="text-xs font-medium text-muted-foreground">
              参考来源 ({message.sources.length})
            </div>
            <div className="grid gap-2">
              {message.sources.map((source, index) => (
                <SourceCard key={source.id || index} source={source} />
              ))}
            </div>
          </div>
        )}

        {/* Message Actions */}
        <div className="flex items-center gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => onCopy(message.content)}
            className="h-6 px-2 text-xs"
          >
            <Copy className="h-3 w-3" />
          </Button>
          
          {!isUser && (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
              >
                <ThumbsUp className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-xs"
              >
                <ThumbsDown className="h-3 w-3" />
              </Button>
            </>
          )}
        </div>

        {/* Timestamp */}
        <div className="text-xs text-muted-foreground mt-1">
          {formatDistanceToNow(message.timestamp, {
            addSuffix: true,
            locale: zhCN,
          })}
        </div>
      </div>
    </div>
  );
}

function SourceCard({ source }: { source: Source }) {
  return (
    <div className="bg-card border border-border rounded-lg p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <h4 className="font-medium truncate">{source.title}</h4>
          <p className="text-muted-foreground text-xs mt-1">
            {source.authors.join(', ')}
          </p>
          {source.published_date && (
            <p className="text-muted-foreground text-xs">
              {source.published_date}
            </p>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          <div className="text-xs bg-muted px-2 py-1 rounded">
            相关度: {(source.relevance_score * 100).toFixed(0)}%
          </div>
          {source.links?.pdf && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => window.open(source.links.pdf, '_blank')}
              className="h-6 px-2"
            >
              <ExternalLink className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>
      
      {source.abstract && (
        <p className="text-muted-foreground text-xs mt-2 line-clamp-2">
          {source.abstract}
        </p>
      )}
    </div>
  );
}