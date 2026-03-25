import { useEffect, useRef, useState } from 'react';
import hljs from 'highlight.js/lib/core';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/github-dark.css';

hljs.registerLanguage('python', python);

interface CodeBlockProps {
  code: string;
  compact?: boolean;
  collapsible?: boolean;
  defaultCollapsed?: boolean;
}

export function CodeBlock({ code, compact = false, collapsible = false, defaultCollapsed = false }: CodeBlockProps) {
  const codeRef = useRef<HTMLElement>(null);
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (codeRef.current && !isCollapsed) {
      codeRef.current.removeAttribute('data-highlighted');
      hljs.highlightElement(codeRef.current);
    }
  }, [code, isCollapsed]);

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
      <div className="relative group">
        <div
          className="bg-surface-sunken border border-surface-border rounded-xl cursor-pointer hover:border-surface-border-light transition-all overflow-hidden"
          onClick={() => setIsCollapsed(false)}
        >
          <div className="flex items-center justify-between text-xs px-4 py-2 bg-surface-overlay border-b border-surface-border">
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4 text-accent-light" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
              <span className="text-text-tertiary">Python</span>
              <span className="text-text-muted">({lineCount} lines)</span>
            </div>
            <span className="flex items-center gap-1 text-accent-light hover:text-accent">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
              Expand
            </span>
          </div>
          <div className="relative">
            <div className="absolute left-0 top-0 bottom-0 w-10 bg-surface-sunken/50 border-r border-surface-border flex flex-col items-end pr-2 pt-3">
              {lines.slice(0, previewLines).map((_, i) => (
                <div key={i} className={`text-text-muted font-mono leading-5 ${compact ? 'text-[10px]' : 'text-xs'}`}>{i + 1}</div>
              ))}
            </div>
            <pre className={`pl-12 pr-4 py-3 text-sand-200 overflow-hidden ${compact ? 'text-xs' : 'text-sm'}`}>
              <code className="language-python">{previewCode}</code>
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
    <div className="relative group">
      <div className="bg-surface-sunken border border-surface-border rounded-xl overflow-hidden">
        <div className="flex items-center justify-between text-xs px-4 py-2 bg-surface-overlay border-b border-surface-border">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-accent-light" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
            </svg>
            <span className="text-text-tertiary">Python</span>
            <span className="text-text-muted">({lineCount} lines)</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 text-text-tertiary hover:text-text-primary transition-colors"
            >
              {copied ? (
                <>
                  <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span className="text-emerald-400">Copied!</span>
                </>
              ) : (
                <>
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Copy
                </>
              )}
            </button>
            {collapsible && (
              <button
                onClick={() => setIsCollapsed(true)}
                className="flex items-center gap-1 text-accent-light hover:text-accent"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                </svg>
                Collapse
              </button>
            )}
          </div>
        </div>
        <div className="relative">
          <div className="absolute left-0 top-0 bottom-0 w-10 bg-surface-sunken/50 border-r border-surface-border flex flex-col items-end pr-2 pt-3 select-none">
            {lines.map((_, i) => (
              <div key={i} className={`text-text-muted font-mono leading-5 ${compact ? 'text-[10px]' : 'text-xs'}`}>{i + 1}</div>
            ))}
          </div>
          <pre className={`pl-12 pr-4 py-3 overflow-x-auto ${compact ? 'text-xs' : 'text-sm'}`}>
            <code ref={codeRef} className="language-python leading-5">
              {code}
            </code>
          </pre>
        </div>
      </div>
    </div>
  );
}
