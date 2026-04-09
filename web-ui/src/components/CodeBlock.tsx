import { useEffect, useState } from 'react';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
// Custom syntax theme in index.css — no base theme import needed

hljs.registerLanguage('python', python);

interface CodeBlockProps {
  code: string;
  language?: string;
  compact?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export function CodeBlock({ code, language = 'python', compact = false, collapsible = false, defaultCollapsed = false }: CodeBlockProps) {
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [copied, setCopied] = useState(false);
  const [highlightedHtml, setHighlightedHtml] = useState<string>(() => {
    if (!defaultCollapsed) {
      return hljs.highlight(code, { language }).value;
    }
    return '';
  });

  useEffect(() => {
    if (!isCollapsed) {
      setHighlightedHtml(hljs.highlight(code, { language }).value);
    }
  }, [code, isCollapsed, language]);

  const lines = code.split('\n');
  const lineCount = lines.length;
  const previewLines = 3;
  const previewCode = lines.slice(0, previewLines).join('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  if (collapsible && isCollapsed) {
    return (
      <div className="relative">
        <div
          className="relative bg-surface-sunken border border-surface-border border-t-2 border-t-accent/20 rounded-md cursor-pointer hover:border-surface-border-light transition-all overflow-hidden"
          onClick={() => setIsCollapsed(false)}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setIsCollapsed(false); } }}
          role="button"
          tabIndex={0}
        >
          {language && (
            <span className="absolute top-2 left-3 text-xs text-accent/40 uppercase tracking-wider select-none z-10">
              {language}
            </span>
          )}
          <div className="absolute top-2 right-2 flex items-center gap-2 z-10">
            <span className="flex items-center gap-1 text-xs text-accent-light hover:text-accent cursor-pointer">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
              Expand
            </span>
          </div>
          <div className="relative pt-8">
            <div className="absolute left-0 top-8 bottom-0 w-10 bg-surface-sunken/50 border-r border-surface-border flex flex-col items-end pr-2 pt-1 select-none">
              {lines.slice(0, previewLines).map((_, i) => (
                <div key={i} className={`text-text-muted font-mono leading-5 ${compact ? 'text-[10px]' : 'text-xs'}`}>{i + 1}</div>
              ))}
            </div>
            <pre className={`pl-12 pr-4 py-1 text-sand-200 overflow-hidden ${compact ? 'text-xs' : 'text-sm'}`}>
              <code className={`language-${language}`}>{previewCode}</code>
            </pre>
          </div>
          {lineCount > previewLines && (
            <div className="px-4 pb-3 text-xs text-text-muted pl-12">... {lineCount - previewLines} more lines</div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      <div className="relative bg-surface-sunken border border-surface-border border-t-2 border-t-accent/20 rounded-md overflow-hidden">
        {language && (
          <span className="absolute top-2 left-3 text-xs text-accent/40 uppercase tracking-wider select-none z-10">
            {language}
          </span>
        )}
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 z-10 p-1.5 rounded text-text-tertiary bg-transparent hover:text-accent hover:bg-surface-overlay transition-colors"
          aria-label="Copy code"
        >
          {copied ? (
            <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          )}
        </button>
        {copied && (
          <span className="absolute top-10 right-2 z-20 text-[10px] text-emerald-400 bg-surface-overlay border border-surface-border rounded px-1.5 py-0.5 shadow-lg">
            Copied!
          </span>
        )}
        <div className="relative pt-8">
          <div className="absolute left-0 top-8 bottom-0 w-10 bg-surface-sunken/50 border-r border-surface-border flex flex-col items-end pr-2 pt-1 select-none">
            {lines.map((_, i) => (
              <div key={i} className={`text-text-muted font-mono leading-5 ${compact ? 'text-[10px]' : 'text-xs'}`}>{i + 1}</div>
            ))}
          </div>
          <pre className={`pl-12 pr-4 py-1 pb-3 overflow-x-auto ${compact ? 'text-xs' : 'text-sm'}`}>
            <code
              className={`language-${language} leading-5`}
              dangerouslySetInnerHTML={{ __html: highlightedHtml }}
            />
          </pre>
        </div>
        {collapsible && (
          <button
            onClick={() => setIsCollapsed(true)}
            className="w-full flex items-center justify-center gap-1 text-xs text-accent-light hover:text-accent py-1.5 border-t border-surface-border bg-surface-overlay/50 transition-colors"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
            </svg>
            Collapse
          </button>
        )}
      </div>
    </div>
  );
}
