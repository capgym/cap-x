import { useState, useEffect, useRef } from 'react';

interface ImageViewerProps {
  src: string;
  alt: string;
  className?: string;
}

export function ImageViewer({ src, alt, className }: ImageViewerProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isExpanded && overlayRef.current) {
      overlayRef.current.focus();
    }
  }, [isExpanded]);

  // Handle both data URIs and plain base64 strings
  const imageSrc = src.startsWith('data:') || src.startsWith('http')
    ? src
    : `data:image/jpeg;base64,${src}`;

  return (
    <>
      <img
        src={imageSrc}
        alt={alt}
        className={className || "rounded-md max-w-full max-h-64 object-contain cursor-pointer hover:opacity-90 transition-opacity border border-surface-border"}
        onClick={() => setIsExpanded(true)}
      />

      {/* Modal for expanded view */}
      {isExpanded && (
        <div
          ref={overlayRef}
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4"
          onClick={() => setIsExpanded(false)}
          onKeyDown={(e) => { if (e.key === 'Escape') setIsExpanded(false); }}
          role="dialog"
          aria-modal="true"
          tabIndex={-1}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]">
            <img
              src={imageSrc}
              alt={alt}
              className="max-w-full max-h-[90vh] object-contain rounded-md"
            />
            <button
              onClick={() => setIsExpanded(false)}
              className="absolute top-2 right-2 w-8 h-8 bg-white/20 hover:bg-white/30 rounded-full flex items-center justify-center text-white transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  );
}
