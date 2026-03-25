import { useState, useEffect, useRef } from 'react';

/**
 * Default Viser URL — goes through the FastAPI reverse proxy so only one
 * port (the web-UI port) needs to be forwarded.
 */
const DEFAULT_VISER_URL = '/viser-proxy/';

export function VisualizationPanel() {
  const [viserUrl, setViserUrl] = useState(DEFAULT_VISER_URL);
  const [isEditing, setIsEditing] = useState(false);
  const [tempUrl, setTempUrl] = useState(viserUrl);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'error'>('connecting');
  const [retryCount, setRetryCount] = useState(0);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const handleSaveUrl = () => {
    setViserUrl(tempUrl);
    setIsEditing(false);
    setConnectionStatus('connecting');
    setRetryCount(0);
  };

  // Poll the viser proxy every 3s. When it transitions from not-ok to ok,
  // bump retryCount to reload the iframe.
  const wasConnectedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      while (!cancelled) {
        try {
          const ctrl = new AbortController();
          const tid = setTimeout(() => ctrl.abort(), 3000);
          const resp = await fetch(viserUrl, { method: 'HEAD', signal: ctrl.signal });
          clearTimeout(tid);

          if (resp.ok) {
            if (!wasConnectedRef.current) {
              // Transition: not connected -> connected — reload iframe
              wasConnectedRef.current = true;
              setRetryCount(c => c + 1);
            }
            setConnectionStatus('connected');
          } else {
            wasConnectedRef.current = false;
            setConnectionStatus('connecting');
          }
        } catch {
          wasConnectedRef.current = false;
          setConnectionStatus('connecting');
        }
        // Wait 3 seconds before next poll
        await new Promise(r => setTimeout(r, 3000));
      }
    }

    poll();
    return () => { cancelled = true; };
  }, [viserUrl]);

  // Handle iframe load event
  const handleIframeLoad = () => {
    setConnectionStatus('connected');
  };

  // Handle iframe error
  const handleIframeError = () => {
    setConnectionStatus('connecting');
  };

  // Manual refresh (just reloads iframe, not the page)
  const handleManualRefresh = () => {
    wasConnectedRef.current = false;
    setRetryCount(c => c + 1);
    setConnectionStatus('connecting');
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 bg-surface-raised border-b border-surface-border flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md bg-surface-overlay flex items-center justify-center">
            <svg className="w-4 h-4 text-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 10l-2 1m0 0l-2-1m2 1v2.5M20 7l-2 1m2-1l-2-1m2 1v2.5M14 4l-2-1-2 1M4 7l2-1M4 7l2 1M4 7v2.5M12 21l-2-1m2 1l2-1m-2 1v-2.5M6 18l-2-1v-2.5M18 18l2-1v-2.5" />
            </svg>
          </div>
          <span className="text-sm font-medium text-text-primary">3D Visualization</span>

          {/* Connection status indicator */}
          <div className="flex items-center gap-1.5 ml-2">
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === 'connected'
                ? 'bg-emerald-400'
                : 'bg-amber-400 animate-pulse'
            }`} />
            <span className="text-xs text-text-tertiary">
              {connectionStatus === 'connected' ? 'Connected' : 'Connecting...'}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isEditing ? (
            <>
              <input
                type="text"
                value={tempUrl}
                onChange={(e) => setTempUrl(e.target.value)}
                className="px-3 py-1.5 text-xs bg-surface-sunken text-text-primary border border-surface-border rounded-lg w-64 focus:ring-2 focus:ring-accent/30 focus:border-accent/40"
                onKeyDown={(e) => e.key === 'Enter' && handleSaveUrl()}
                autoFocus
              />
              <button
                onClick={handleSaveUrl}
                className="px-3 py-1.5 text-xs bg-accent text-white rounded-lg hover:bg-accent-dark transition-colors"
              >
                Save
              </button>
              <button
                onClick={() => {
                  setTempUrl(viserUrl);
                  setIsEditing(false);
                }}
                className="px-3 py-1.5 text-xs bg-surface-overlay text-text-secondary rounded-lg hover:bg-surface-border transition-colors"
              >
                Cancel
              </button>
            </>
          ) : (
            <>
              <span className="text-xs text-text-tertiary max-w-[200px] truncate" title={viserUrl}>{viserUrl}</span>
              <button
                onClick={handleManualRefresh}
                className="p-1.5 text-text-tertiary hover:text-text-primary hover:bg-surface-overlay rounded-lg transition-colors"
                title="Refresh connection"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
              <button
                onClick={() => setIsEditing(true)}
                className="p-1.5 text-text-tertiary hover:text-text-primary hover:bg-surface-overlay rounded-lg transition-colors"
                title="Edit URL"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              </button>
              <button
                onClick={() => window.open(viserUrl, '_blank')}
                className="p-1.5 text-text-tertiary hover:text-text-primary hover:bg-surface-overlay rounded-lg transition-colors"
                title="Open in new tab"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Iframe Container */}
      <div className="flex-1 relative bg-surface-sunken">
        {connectionStatus !== 'connected' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-surface-sunken/80">
            <div className="flex items-center gap-3 text-text-tertiary mb-2">
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              <span>Waiting for Viser server...</span>
            </div>
            <span className="text-xs text-text-muted">Auto-retrying every 3 seconds</span>
          </div>
        )}
        <iframe
          ref={iframeRef}
          key={`viser-${retryCount}`}
          src={viserUrl}
          title="Viser 3D Visualization"
          className="absolute inset-0 w-full h-full border-0"
          allow="autoplay; fullscreen"
          onLoad={handleIframeLoad}
          onError={handleIframeError}
        />
      </div>
    </div>
  );
}
