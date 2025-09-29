'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, Calendar, Users, ExternalLink, Star, Eye, Download } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import Loading from '@/components/ui/Loading';
import { TrendingPaper } from '@/types';
import { cn, formatNumber } from '@/lib/utils';

export default function TrendingInterface() {
  const { trendingPapers, setTrendingPapers } = useAppStore();
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<7 | 30 | 90>(7);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  useEffect(() => {
    loadTrendingPapers();
  }, [timeRange]);

  const loadTrendingPapers = async () => {
    setLoading(true);
    try {
      const papers = await apiClient.getTrendingPapers(timeRange, 50);
      setTrendingPapers(papers);
    } catch (error) {
      console.error('Failed to load trending papers:', error);
    } finally {
      setLoading(false);
    }
  };

  const categories = [
    { value: 'all', label: '全部' },
    { value: 'cs.AI', label: '人工智能' },
    { value: 'cs.CL', label: '计算语言学' },
    { value: 'cs.CV', label: '计算机视觉' },
    { value: 'cs.LG', label: '机器学习' },
    { value: 'cs.IR', label: '信息检索' },
    { value: 'stat.ML', label: '统计学习' },
  ];

  const filteredPapers = selectedCategory === 'all' 
    ? trendingPapers 
    : trendingPapers.filter(paper => 
        paper.categories.some(cat => cat.includes(selectedCategory))
      );

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-6 w-6 text-primary" />
              <h1 className="text-2xl font-bold">热门论文</h1>
            </div>
            <Button
              onClick={loadTrendingPapers}
              disabled={loading}
              size="sm"
              className="gap-2"
            >
              {loading ? '刷新中...' : '刷新'}
            </Button>
          </div>

          <div className="flex flex-col sm:flex-row gap-4">
            {/* Time Range Selector */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">时间范围:</span>
              <div className="flex bg-muted rounded-lg p-1">
                {[
                  { value: 7, label: '7天' },
                  { value: 30, label: '30天' },
                  { value: 90, label: '90天' },
                ].map((option) => (
                  <button
                    key={option.value}
                    onClick={() => setTimeRange(option.value as any)}
                    className={cn(
                      'px-3 py-1 text-sm font-medium rounded transition-colors',
                      timeRange === option.value 
                        ? 'bg-background shadow-sm' 
                        : 'hover:bg-background/50'
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Category Filter */}
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">分类:</span>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="px-3 py-1 text-sm bg-background border border-border rounded focus:outline-none focus:ring-2 focus:ring-ring"
              >
                {categories.map((category) => (
                  <option key={category.value} value={category.value}>
                    {category.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto p-4">
          {loading ? (
            <div className="h-64 flex items-center justify-center">
              <Loading size="lg" text="加载热门论文中..." />
            </div>
          ) : filteredPapers.length === 0 ? (
            <div className="h-64 flex flex-col items-center justify-center text-center">
              <TrendingUp className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">暂无热门论文</h3>
              <p className="text-muted-foreground">
                {selectedCategory === 'all' 
                  ? '最近没有发现热门论文，请稍后再试'
                  : '该分类下暂无热门论文，尝试选择其他分类'
                }
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  共 {formatNumber(filteredPapers.length)} 篇热门论文
                </p>
              </div>

              <div className="grid gap-6">
                {filteredPapers.map((paper, index) => (
                  <TrendingPaperCard
                    key={paper.id}
                    paper={paper}
                    rank={index + 1}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TrendingPaperCard({ 
  paper, 
  rank 
}: { 
  paper: TrendingPaper; 
  rank: number; 
}) {
  const trendingScore = paper.trending_score || 0;
  
  const getRankColor = (rank: number) => {
    if (rank <= 3) return 'text-yellow-600 bg-yellow-100';
    if (rank <= 10) return 'text-blue-600 bg-blue-100';
    return 'text-gray-600 bg-gray-100';
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'text-red-600 bg-red-100';
    if (score >= 0.6) return 'text-orange-600 bg-orange-100';
    if (score >= 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-green-600 bg-green-100';
  };

  return (
    <div className="bg-card border border-border rounded-lg p-6 hover:shadow-md transition-all">
      <div className="flex items-start gap-4">
        {/* Rank Badge */}
        <div className={cn(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold',
          getRankColor(rank)
        )}>
          {rank}
        </div>

        <div className="flex-1 min-w-0 space-y-4">
          {/* Title and Score */}
          <div className="flex items-start justify-between gap-4">
            <h3 className="font-semibold text-lg leading-tight hover:text-primary cursor-pointer">
              {paper.title}
            </h3>
            <div className="flex items-center gap-2">
              <div className={cn(
                'px-2 py-1 rounded-full text-xs font-medium',
                getScoreColor(trendingScore)
              )}>
                热度 {(trendingScore * 100).toFixed(0)}
              </div>
              {paper.url && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => window.open(paper.url, '_blank')}
                  className="p-1"
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          {/* Metadata */}
          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            {paper.authors.length > 0 && (
              <div className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                <span>{paper.authors.slice(0, 4).join(', ')}</span>
                {paper.authors.length > 4 && <span> 等</span>}
              </div>
            )}

            {paper.published_date && (
              <div className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                <span>{new Date(paper.published_date).toLocaleDateString('zh-CN')}</span>
              </div>
            )}
          </div>

          {/* Abstract */}
          {paper.abstract && (
            <p className="text-sm text-muted-foreground line-clamp-4 leading-relaxed">
              {paper.abstract}
            </p>
          )}

          {/* Categories and Actions */}
          <div className="flex items-center justify-between">
            <div className="flex flex-wrap gap-2">
              {paper.categories.slice(0, 3).map((category, index) => (
                <span
                  key={index}
                  className="text-xs px-2 py-1 bg-muted rounded-full"
                >
                  {category}
                </span>
              ))}
              {paper.categories.length > 3 && (
                <span className="text-xs px-2 py-1 bg-muted rounded-full">
                  +{paper.categories.length - 3}
                </span>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
              >
                <Star className="h-3 w-3" />
                收藏
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="gap-2"
              >
                <Download className="h-3 w-3" />
                下载
              </Button>
            </div>
          </div>

          {/* Trending Indicators */}
          <div className="flex items-center gap-4 pt-2 border-t border-border">
            <div className="flex items-center gap-1 text-xs text-muted-foreground">
              <Eye className="h-3 w-3" />
              <span>热度趋势</span>
            </div>
            
            {/* Simple trending visualization */}
            <div className="flex-1 flex items-center gap-1">
              {Array.from({ length: 7 }, (_, i) => (
                <div
                  key={i}
                  className={cn(
                    'w-1 rounded-full transition-all',
                    i < Math.floor(trendingScore * 7) 
                      ? 'bg-primary h-3' 
                      : 'bg-muted h-1'
                  )}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}