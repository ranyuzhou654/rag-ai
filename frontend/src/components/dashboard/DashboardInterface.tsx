'use client';

import { useState, useEffect } from 'react';
import { 
  BarChart3, 
  Activity, 
  Database, 
  Zap, 
  Users, 
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw
} from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import Loading from '@/components/ui/Loading';
import { SystemStats, CacheLayerStats } from '@/types';
import { cn, formatNumber, formatFileSize } from '@/lib/utils';

export default function DashboardInterface() {
  const { 
    systemStats, 
    setSystemStats, 
    healthStatus, 
    setHealthStatus 
  } = useAppStore();

  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [stats, health] = await Promise.all([
        apiClient.getSystemStats(),
        apiClient.getHealthStatus(),
      ]);
      
      setSystemStats(stats);
      setHealthStatus(health.status as any);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      setHealthStatus('critical');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !systemStats) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loading size="lg" text="加载系统监控数据..." />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-bold">系统监控</h1>
            </div>
            
            <div className="flex items-center gap-4">
              {lastUpdate && (
                <div className="text-sm text-muted-foreground">
                  最后更新: {lastUpdate.toLocaleTimeString('zh-CN')}
                </div>
              )}
              <Button
                onClick={loadDashboardData}
                disabled={loading}
                size="sm"
                className="gap-2"
              >
                <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
                刷新
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto p-4 space-y-6">
          {/* System Health Overview */}
          <SystemHealthOverview healthStatus={healthStatus} />

          {/* Stats Grid */}
          {systemStats && (
            <>
              <CacheStatsGrid cacheStats={systemStats.cache} />
              <CitationStatsGrid citationStats={systemStats.citations} />
              <FeedbackStatsGrid feedbackStats={systemStats.feedback} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function SystemHealthOverview({ 
  healthStatus 
}: { 
  healthStatus: 'healthy' | 'warning' | 'critical' | 'unknown'; 
}) {
  const getHealthConfig = () => {
    switch (healthStatus) {
      case 'healthy':
        return {
          icon: CheckCircle,
          color: 'text-green-600',
          bgColor: 'bg-green-100',
          status: '系统正常',
          description: '所有服务运行正常',
        };
      case 'warning':
        return {
          icon: AlertTriangle,
          color: 'text-yellow-600',
          bgColor: 'bg-yellow-100',
          status: '系统警告',
          description: '检测到性能问题',
        };
      case 'critical':
        return {
          icon: XCircle,
          color: 'text-red-600',
          bgColor: 'bg-red-100',
          status: '系统故障',
          description: '存在严重问题需要处理',
        };
      default:
        return {
          icon: Activity,
          color: 'text-gray-600',
          bgColor: 'bg-gray-100',
          status: '状态未知',
          description: '无法获取系统状态',
        };
    }
  };

  const config = getHealthConfig();
  const Icon = config.icon;

  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <div className="flex items-center gap-4">
        <div className={cn('p-3 rounded-full', config.bgColor)}>
          <Icon className={cn('h-8 w-8', config.color)} />
        </div>
        
        <div className="flex-1">
          <h2 className="text-xl font-semibold">{config.status}</h2>
          <p className="text-muted-foreground">{config.description}</p>
        </div>

        <div className="text-right">
          <div className="text-2xl font-bold">{healthStatus.toUpperCase()}</div>
          <div className="text-sm text-muted-foreground">
            {new Date().toLocaleString('zh-CN')}
          </div>
        </div>
      </div>
    </div>
  );
}

function CacheStatsGrid({ 
  cacheStats 
}: { 
  cacheStats: SystemStats['cache']; 
}) {
  const layers = [
    { key: 'memory', name: '内存缓存', icon: Zap },
    { key: 'redis', name: 'Redis缓存', icon: Database },
    { key: 'file', name: '文件缓存', icon: Clock },
    { key: 'vector', name: '向量缓存', icon: Activity },
  ];

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">缓存系统性能</h2>
      
      {/* Overall Stats */}
      <div className="bg-card border border-border rounded-lg p-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">
              {formatNumber(cacheStats.overall.total_requests)}
            </div>
            <div className="text-sm text-muted-foreground">总请求数</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {formatNumber(cacheStats.overall.total_hits)}
            </div>
            <div className="text-sm text-muted-foreground">缓存命中</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {(cacheStats.overall.hit_rate * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">命中率</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {Object.keys(cacheStats.layers).length}
            </div>
            <div className="text-sm text-muted-foreground">缓存层数</div>
          </div>
        </div>
      </div>

      {/* Layer Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {layers.map(({ key, name, icon: Icon }) => {
          const layerStats = cacheStats.layers[key as keyof typeof cacheStats.layers];
          if (!layerStats) return null;

          return (
            <CacheLayerCard
              key={key}
              name={name}
              icon={Icon}
              stats={layerStats}
            />
          );
        })}
      </div>
    </div>
  );
}

function CacheLayerCard({ 
  name, 
  icon: Icon, 
  stats 
}: { 
  name: string; 
  icon: any; 
  stats: CacheLayerStats; 
}) {
  return (
    <div className="bg-card border border-border rounded-lg p-4">
      <div className="flex items-center gap-3 mb-3">
        <Icon className="h-5 w-5 text-primary" />
        <h3 className="font-medium">{name}</h3>
      </div>
      
      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-sm text-muted-foreground">命中率</span>
          <span className="text-sm font-medium">
            {(stats.hit_rate * 100).toFixed(1)}%
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-sm text-muted-foreground">命中数</span>
          <span className="text-sm font-medium">
            {formatNumber(stats.hits)}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-sm text-muted-foreground">未命中</span>
          <span className="text-sm font-medium">
            {formatNumber(stats.misses)}
          </span>
        </div>

        {stats.total_size_bytes && (
          <div className="flex justify-between">
            <span className="text-sm text-muted-foreground">大小</span>
            <span className="text-sm font-medium">
              {formatFileSize(stats.total_size_bytes)}
            </span>
          </div>
        )}

        {stats.connected !== undefined && (
          <div className="flex items-center gap-2">
            <div className={cn(
              'w-2 h-2 rounded-full',
              stats.connected ? 'bg-green-500' : 'bg-red-500'
            )} />
            <span className="text-sm text-muted-foreground">
              {stats.connected ? '已连接' : '未连接'}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function CitationStatsGrid({ 
  citationStats 
}: { 
  citationStats: SystemStats['citations']; 
}) {
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">引用统计</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <Database className="h-5 w-5 text-blue-600" />
            <h3 className="font-medium">总文档数</h3>
          </div>
          <div className="text-2xl font-bold">
            {formatNumber(citationStats.total_sources)}
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="h-5 w-5 text-green-600" />
            <h3 className="font-medium">总引用数</h3>
          </div>
          <div className="text-2xl font-bold">
            {formatNumber(citationStats.total_citations)}
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <BarChart3 className="h-5 w-5 text-purple-600" />
            <h3 className="font-medium">平均引用</h3>
          </div>
          <div className="text-2xl font-bold">
            {citationStats.average_citations_per_source.toFixed(1)}
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <Activity className="h-5 w-5 text-orange-600" />
            <h3 className="font-medium">数据源类型</h3>
          </div>
          <div className="text-2xl font-bold">
            {Object.keys(citationStats.source_types).length}
          </div>
        </div>
      </div>

      {/* Source Types Breakdown */}
      <div className="bg-card border border-border rounded-lg p-4">
        <h3 className="font-medium mb-3">数据源分布</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(citationStats.source_types).map(([type, count]) => (
            <div key={type} className="text-center">
              <div className="text-lg font-semibold">{formatNumber(count)}</div>
              <div className="text-sm text-muted-foreground capitalize">{type}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function FeedbackStatsGrid({ 
  feedbackStats 
}: { 
  feedbackStats: SystemStats['feedback']; 
}) {
  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">用户反馈</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <Users className="h-5 w-5 text-blue-600" />
            <h3 className="font-medium">总反馈数</h3>
          </div>
          <div className="text-2xl font-bold">
            {formatNumber(feedbackStats.total_feedback)}
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <TrendingUp className="h-5 w-5 text-green-600" />
            <h3 className="font-medium">平均评分</h3>
          </div>
          <div className="text-2xl font-bold">
            {feedbackStats.average_rating.toFixed(1)}/5.0
          </div>
        </div>

        <div className="bg-card border border-border rounded-lg p-4">
          <div className="flex items-center gap-3 mb-2">
            <Clock className="h-5 w-5 text-purple-600" />
            <h3 className="font-medium">最近反馈</h3>
          </div>
          <div className="text-2xl font-bold">
            {feedbackStats.recent_feedback.length}
          </div>
        </div>
      </div>
    </div>
  );
}