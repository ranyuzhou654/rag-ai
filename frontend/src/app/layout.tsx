import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'RAG-AI | 智能检索增强生成系统',
  description: '基于最新技术的中文学术文档智能问答系统，支持多模态检索、知识图谱和分层生成',
  keywords: ['RAG', '人工智能', '检索增强生成', '学术搜索', '知识图谱'],
  authors: [{ name: 'RAG-AI Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className={inter.className} suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}