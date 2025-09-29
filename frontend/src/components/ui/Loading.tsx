'use client';

import { cn } from '@/lib/utils';
import { LoadingProps } from '@/types';

export default function Loading({ className, size = 'md', text, children }: LoadingProps) {
  const sizes = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
  };

  return (
    <div className={cn('flex flex-col items-center justify-center gap-2', className)}>
      <div className={cn(
        'animate-spin rounded-full border-2 border-muted border-t-primary',
        sizes[size]
      )} />
      {(text || children) && (
        <div className="text-sm text-muted-foreground">
          {text || children}
        </div>
      )}
    </div>
  );
}