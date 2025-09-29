'use client';

import { useState, useEffect } from 'react';
import { Search, Filter, Download, ExternalLink, Calendar, Users, Tag } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { apiClient } from '@/lib/api';
import { Button } from '@/components/ui/Button';
import Loading from '@/components/ui/Loading';
import { Document, SearchFilters } from '@/types';
import { cn, formatNumber, debounce } from '@/lib/utils';

export default function SearchInterface() {
  const { 
    searchResults, 
    searchLoading, 
    setSearchResults, 
    setSearchLoading,
    preferences 
  } = useAppStore();

  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({});
  const [showFilters, setShowFilters] = useState(false);
  const [searchType, setSearchType] = useState<'semantic' | 'keyword' | 'hybrid' | 'academic'>('hybrid');
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());

  const debouncedSearch = debounce(performSearch, 500);

  useEffect(() => {
    if (query.trim()) {
      debouncedSearch();
    } else {
      setSearchResults([]);
    }
  }, [query, searchType, filters]);

  async function performSearch() {
    if (!query.trim()) return;

    setSearchLoading(true);
    try {
      const response = await apiClient.searchDocuments({
        query: query.trim(),
        limit: 50,
        filters,
        search_type: searchType,
      });
      setSearchResults(response.results);
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
    } finally {
      setSearchLoading(false);
    }
  }

  const handleDocumentSelect = (documentId: string) => {
    const newSelected = new Set(selectedDocuments);
    if (newSelected.has(documentId)) {
      newSelected.delete(documentId);
    } else {
      newSelected.add(documentId);
    }
    setSelectedDocuments(newSelected);
  };

  const handleExportSelected = async () => {
    if (selectedDocuments.size === 0) return;

    try {
      const bibliography = await apiClient.exportBibliography(
        Array.from(selectedDocuments),
        preferences.citationFormat
      );
      
      const blob = new Blob([bibliography], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `bibliography.${preferences.citationFormat}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export error:', error);
    }
  };

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Search Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm p-4">
        <div className="max-w-6xl mx-auto space-y-4">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-muted-foreground" />
            <input
              type="text"
              placeholder="搜索学术文档、论文、技术博客..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-3 text-sm bg-background border border-border rounded-lg focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent"
            />
          </div>

          {/* Search Options */}
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">搜索模式:</span>
              <select
                value={searchType}
                onChange={(e) => setSearchType(e.target.value as any)}
                className="text-sm bg-background border border-border rounded px-3 py-1 focus:outline-none focus:ring-2 focus:ring-ring"
              >
                <option value="hybrid">混合搜索</option>
                <option value="semantic">语义搜索</option>
                <option value="keyword">关键词搜索</option>
                <option value="academic">学术搜索</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowFilters(!showFilters)}
                className="gap-2"
              >
                <Filter className="h-4 w-4" />
                筛选
              </Button>

              {selectedDocuments.size > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleExportSelected}
                  className="gap-2"
                >
                  <Download className="h-4 w-4" />
                  导出 ({selectedDocuments.size})
                </Button>
              )}
            </div>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <SearchFiltersPanel 
              filters={filters} 
              onChange={setFilters} 
            />
          )}
        </div>
      </div>

      {/* Results Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-6xl mx-auto p-4">
          {searchLoading ? (
            <div className="h-64 flex items-center justify-center">
              <Loading size="lg" text="搜索中..." />
            </div>
          ) : query && searchResults.length === 0 ? (
            <div className="h-64 flex flex-col items-center justify-center text-center">
              <Search className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">未找到相关文档</h3>
              <p className="text-muted-foreground">尝试调整搜索关键词或筛选条件</p>
            </div>
          ) : searchResults.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  找到 {formatNumber(searchResults.length)} 个相关文档
                </p>
              </div>

              <div className="grid gap-4">
                {searchResults.map((document) => (
                  <DocumentCard
                    key={document.id}
                    document={document}
                    selected={selectedDocuments.has(document.id)}
                    onSelect={() => handleDocumentSelect(document.id)}
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center text-center">
              <Search className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">开始搜索</h3>
              <p className="text-muted-foreground mb-4">
                在上方输入框中输入关键词来搜索学术文档
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-md">
                {[
                  'Transformer架构原理',
                  '大模型微调技术',
                  'RAG系统优化',
                  '向量数据库比较'
                ].map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => setQuery(suggestion)}
                    className="p-2 text-left border border-border rounded hover:bg-accent transition-colors text-sm"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function SearchFiltersPanel({ 
  filters, 
  onChange 
}: { 
  filters: SearchFilters; 
  onChange: (filters: SearchFilters) => void; 
}) {
  return (
    <div className="bg-muted/50 rounded-lg p-4 space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Year Range */}
        <div>
          <label className="block text-sm font-medium mb-2">发表年份</label>
          <div className="flex gap-2">
            <input
              type="number"
              placeholder="从"
              min="1990"
              max={new Date().getFullYear()}
              value={filters.year_range?.[0] || ''}
              onChange={(e) => onChange({
                ...filters,
                year_range: [parseInt(e.target.value) || new Date().getFullYear() - 10, filters.year_range?.[1] || new Date().getFullYear()]
              })}
              className="flex-1 px-2 py-1 text-sm border border-border rounded focus:outline-none focus:ring-1 focus:ring-ring"
            />
            <input
              type="number"
              placeholder="到"
              min="1990"
              max={new Date().getFullYear()}
              value={filters.year_range?.[1] || ''}
              onChange={(e) => onChange({
                ...filters,
                year_range: [filters.year_range?.[0] || 2020, parseInt(e.target.value) || new Date().getFullYear()]
              })}
              className="flex-1 px-2 py-1 text-sm border border-border rounded focus:outline-none focus:ring-1 focus:ring-ring"
            />
          </div>
        </div>

        {/* Sources */}
        <div>
          <label className="block text-sm font-medium mb-2">数据源</label>
          <select
            multiple
            value={filters.sources || []}
            onChange={(e) => onChange({
              ...filters,
              sources: Array.from(e.target.selectedOptions, option => option.value)
            })}
            className="w-full px-2 py-1 text-sm border border-border rounded focus:outline-none focus:ring-1 focus:ring-ring"
            size={3}
          >
            <option value="arxiv">arXiv</option>
            <option value="huggingface">HuggingFace</option>
            <option value="blog">技术博客</option>
            <option value="journal">期刊论文</option>
          </select>
        </div>

        {/* Categories */}
        <div>
          <label className="block text-sm font-medium mb-2">主题分类</label>
          <input
            type="text"
            placeholder="输入分类标签，用逗号分隔"
            value={filters.categories?.join(', ') || ''}
            onChange={(e) => onChange({
              ...filters,
              categories: e.target.value.split(',').map(c => c.trim()).filter(Boolean)
            })}
            className="w-full px-2 py-1 text-sm border border-border rounded focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>
      </div>

      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={filters.has_full_text || false}
            onChange={(e) => onChange({
              ...filters,
              has_full_text: e.target.checked
            })}
            className="rounded border-border focus:ring-ring"
          />
          仅显示全文可用的文档
        </label>
      </div>
    </div>
  );
}

function DocumentCard({ 
  document, 
  selected, 
  onSelect 
}: { 
  document: Document; 
  selected: boolean; 
  onSelect: () => void; 
}) {
  return (
    <div className={cn(
      'bg-card border border-border rounded-lg p-4 transition-all hover:shadow-md',
      selected && 'ring-2 ring-primary bg-primary/5'
    )}>
      <div className="flex items-start gap-4">
        <input
          type="checkbox"
          checked={selected}
          onChange={onSelect}
          className="mt-1 rounded border-border focus:ring-ring"
        />

        <div className="flex-1 min-w-0 space-y-3">
          {/* Title and Actions */}
          <div className="flex items-start justify-between gap-4">
            <h3 className="font-semibold text-lg leading-tight hover:text-primary cursor-pointer">
              {document.title}
            </h3>
            <div className="flex items-center gap-2">
              {document.url && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => window.open(document.url, '_blank')}
                  className="p-1"
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          {/* Metadata */}
          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            {document.authors.length > 0 && (
              <div className="flex items-center gap-1">
                <Users className="h-3 w-3" />
                <span>{document.authors.slice(0, 3).join(', ')}</span>
                {document.authors.length > 3 && <span> 等</span>}
              </div>
            )}

            {document.published_date && (
              <div className="flex items-center gap-1">
                <Calendar className="h-3 w-3" />
                <span>{document.published_date}</span>
              </div>
            )}

            {document.categories.length > 0 && (
              <div className="flex items-center gap-1">
                <Tag className="h-3 w-3" />
                <span>{document.categories[0]}</span>
                {document.categories.length > 1 && (
                  <span className="text-xs bg-muted px-1 rounded">
                    +{document.categories.length - 1}
                  </span>
                )}
              </div>
            )}
          </div>

          {/* Abstract */}
          {document.abstract && (
            <p className="text-sm text-muted-foreground line-clamp-3">
              {document.abstract}
            </p>
          )}

          {/* Source Badge */}
          <div className="flex items-center gap-2">
            <span className={cn(
              'text-xs px-2 py-1 rounded-full font-medium',
              document.source === 'arxiv' && 'bg-red-100 text-red-800',
              document.source === 'huggingface' && 'bg-yellow-100 text-yellow-800',
              document.source === 'blog' && 'bg-blue-100 text-blue-800',
              document.source === 'journal' && 'bg-green-100 text-green-800'
            )}>
              {document.source}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}