import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import type { ExecutionStepData } from '../types/messages';
import { ImageViewer } from './ImageViewer';

interface ExecutionDetailDropdownProps {
  steps: ExecutionStepData[];
  blockIndex: number;
  isExecuting?: boolean;
}

function ImageGrid({ images, maxVisible = 4 }: { images: string[]; maxVisible?: number }) {
  const [showAll, setShowAll] = useState(false);
  const visibleImages = showAll ? images : images.slice(0, maxVisible);
  const hiddenCount = images.length - maxVisible;

  if (images.length === 0) return null;

  return (
    <div className="mt-2">
      <div className="flex flex-wrap gap-2">
        {visibleImages.map((img, idx) => (
          <div key={idx} className="relative group">
            <ImageViewer
              src={img}
              alt={`Step image ${idx + 1}`}
              className="h-36 w-auto rounded-lg border border-surface-border hover:border-surface-border-light transition-colors cursor-pointer"
            />
          </div>
        ))}
      </div>
      {!showAll && hiddenCount > 0 && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-2 text-xs text-accent hover:text-accent-dark transition-colors flex items-center gap-1"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
          Show {hiddenCount} more image{hiddenCount > 1 ? 's' : ''}
        </button>
      )}
      {showAll && hiddenCount > 0 && (
        <button
          onClick={() => setShowAll(false)}
          className="mt-2 text-xs text-accent hover:text-accent-dark transition-colors flex items-center gap-1"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
          </svg>
          Show less
        </button>
      )}
    </div>
  );
}

function ExecutionStep({ step }: { step: ExecutionStepData }) {
  const isHighlighted = step.highlight === true;

  return (
    <div className={`border-l-2 pl-3 py-2 ${
      isHighlighted
        ? 'border-amber-400 bg-amber-900/20 rounded-r-lg'
        : 'border-accent/30'
    }`}>
      <div className="flex items-center gap-2 mb-1">
        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
          isHighlighted
            ? 'bg-amber-900/30 text-amber-400 border border-amber-800/50'
            : 'bg-accent/10 text-accent'
        }`}>
          {step.toolName}
        </span>
        <span className="text-text-tertiary text-xs">Step {step.stepIndex + 1}</span>
      </div>
      <div className={`text-sm prose prose-sm max-w-none prose-warm ${
        isHighlighted ? 'text-amber-300' : 'text-text-secondary'
      }`}>
        <ReactMarkdown>{step.text}</ReactMarkdown>
      </div>
      {step.images.length > 0 && (
        <ImageGrid images={step.images} maxVisible={4} />
      )}
    </div>
  );
}

export function ExecutionDetailDropdown({ steps, blockIndex, isExecuting }: ExecutionDetailDropdownProps) {
  const [expanded, setExpanded] = useState(true);

  if (steps.length === 0) return null;

  return (
    <div className="mt-2 bg-surface-raised/50 rounded-lg border border-surface-border overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-3 py-2 flex items-center justify-between text-left hover:bg-surface-overlay transition-colors"
      >
        <div className="flex items-center gap-2">
          <svg className={`w-4 h-4 text-accent transition-transform ${expanded ? 'rotate-90' : ''}`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
          </svg>
          <span className="text-sm font-medium text-accent">
            Execution Details
          </span>
          <span className="text-xs text-text-tertiary">
            ({steps.length} step{steps.length !== 1 ? 's' : ''})
          </span>
          {isExecuting && (
            <svg className="w-3 h-3 text-accent animate-spin" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
          )}
        </div>
        <span className="text-xs text-text-tertiary">Block {blockIndex + 1}</span>
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-1 space-y-2">
          {steps.map((step, idx) => (
            <ExecutionStep key={idx} step={step} />
          ))}
        </div>
      )}
    </div>
  );
}
