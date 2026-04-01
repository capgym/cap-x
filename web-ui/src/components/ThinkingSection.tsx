import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

interface ThinkingSectionProps {
  content: string;
}

export function ThinkingSection({ content }: ThinkingSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const cleanContent = content
    .replace(/<thinking>/gi, '')
    .replace(/<\/thinking>/gi, '')
    .trim();

  if (!cleanContent) return null;

  const wordCount = cleanContent.split(/\s+/).length;

  return (
    <div className="mb-3">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm text-text-tertiary hover:text-text-primary transition-colors"
      >
        <svg className={`w-3 h-3 transition-transform ${isExpanded ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
        </svg>
        <svg className="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
        </svg>
        <span className="font-medium">Reasoning</span>
        <span className="text-text-tertiary">({wordCount} words)</span>
      </button>

      {isExpanded && (
        <div className="mt-2 ml-5 pl-3 border-l-2 border-accent/30 bg-surface-raised/50 rounded-r-md p-3">
          <div className="text-sm text-text-secondary leading-relaxed prose prose-sm max-w-none prose-invert">
            <ReactMarkdown>{cleanContent}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
