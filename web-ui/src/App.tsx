import { useState, useEffect, useRef, useCallback } from 'react';
import { useTrialState } from './hooks/useTrialState';
import { ConfigStartControl } from './components/ConfigStartControl';
import { ChatPanel } from './components/ChatPanel';
import { VisualizationPanel } from './components/VisualizationPanel';

const FALLBACK_CONFIG = 'env_configs/cube_stack/franka_robosuite_cube_stack.yaml';

function App() {
  const trial = useTrialState();
  const [model, setModel] = useState('aws/anthropic/claude-opus-4-5');
  const [serverUrl, setServerUrl] = useState('http://127.0.0.1:8110/chat/completions');
  const [temperature, setTemperature] = useState(1.0);
  const [awaitUserInput, setAwaitUserInput] = useState(true);
  const [hasCheckedSession, setHasCheckedSession] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Resizable panel
  const [splitPercent, setSplitPercent] = useState(60);
  const draggingRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Ref to track whether we should auto-start after config loads
  const autoStartRef = useRef(false);

  // Close settings popover when clicking outside
  const settingsRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setShowSettings(false);
      }
    }
    if (showSettings) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showSettings]);

  // Check for active session on mount, then load default config
  useEffect(() => {
    const init = async () => {
      const activeSession = await trial.checkActiveSession();
      if (activeSession?.session_id) {
        trial.reconnectToSession(
          activeSession.session_id,
          activeSession.config_path || FALLBACK_CONFIG
        );
        if (activeSession.config_path) {
          trial.loadConfig(activeSession.config_path);
        }
        setHasCheckedSession(true);
        return;
      }

      try {
        const resp = await fetch('/api/default-config');
        const data = await resp.json();
        const configPath = data.config_path || FALLBACK_CONFIG;
        const shouldAutoStart = data.auto_start === true;

        const loaded = await trial.loadConfig(configPath);

        if (shouldAutoStart && loaded) {
          autoStartRef.current = true;
        }
      } catch {
        trial.loadConfig(FALLBACK_CONFIG);
      }
      setHasCheckedSession(true);
    };
    init();
  }, []);

  // Auto-start trial once config is loaded and flag is set
  useEffect(() => {
    if (autoStartRef.current && trial.configPath && trial.state === 'idle' && hasCheckedSession) {
      autoStartRef.current = false;
      trial.startTrial({
        config_path: trial.configPath,
        model,
        server_url: serverUrl,
        temperature,
        await_user_input_each_turn: awaitUserInput,
      });
    }
  }, [trial.configPath, trial.state, hasCheckedSession]);

  const isRunning = trial.state === 'running' || trial.state === 'awaiting_user_input';

  // Drag handler for resizable split
  const [isDragging, setIsDragging] = useState(false);
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    setIsDragging(true);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const handleMouseMove = (e: MouseEvent) => {
      if (!draggingRef.current || !containerRef.current) return;
      e.preventDefault();
      const rect = containerRef.current.getBoundingClientRect();
      const pct = ((e.clientX - rect.left) / rect.width) * 100;
      setSplitPercent(Math.min(70, Math.max(30, pct)));
    };

    const handleMouseUp = () => {
      draggingRef.current = false;
      setIsDragging(false);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, []);

  if (!hasCheckedSession) {
    return (
      <div className="h-full flex items-center justify-center bg-surface">
        <div className="flex items-center gap-3 text-text-tertiary">
          <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          <span className="text-sm font-medium">Initializing...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-surface">
      {/* Header */}
      <header className="flex-shrink-0 bg-surface-raised border-b border-surface-border">
        <div className="flex items-center h-14 px-4 gap-4">
          {/* Logo */}
          <span className="text-lg font-semibold text-text-primary flex-shrink-0">CaP-X</span>

          {/* Divider */}
          <div className="h-6 w-px bg-surface-border flex-shrink-0" />

          {/* Config + Start Control (center) */}
          <div className="flex-1 flex justify-center">
            <ConfigStartControl
              state={trial.state}
              configPath={trial.configPath}
              error={trial.error}
              loadConfig={trial.loadConfig}
              startTrial={trial.startTrial}
              stopTrial={trial.stopTrial}
              reset={trial.reset}
              model={model}
              serverUrl={serverUrl}
              temperature={temperature}
              awaitUserInput={awaitUserInput}
            />
          </div>

          {/* Right side: settings gear + status */}
          <div className="flex items-center gap-3 flex-shrink-0">
            {/* Status indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-surface-overlay">
              <div className={`w-2 h-2 rounded-full ${
                trial.isConnected
                  ? isRunning
                    ? 'bg-amber-400 animate-pulse'
                    : 'bg-emerald-400'
                  : 'bg-sand-500'
              }`} />
              <span className="text-xs text-text-secondary font-medium">
                {isRunning ? 'Running' : trial.isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>

            {/* Settings Gear */}
            <div className="relative" ref={settingsRef}>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-1.5 rounded-lg transition-colors ${
                showSettings
                  ? 'bg-surface-overlay text-text-primary'
                  : 'text-text-tertiary hover:text-text-primary hover:bg-surface-overlay'
              }`}
              title="Settings"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>

            {/* Settings Popover */}
            {showSettings && (
              <div className="absolute right-0 top-full mt-2 w-96 bg-surface-overlay rounded-xl shadow-xl border border-surface-border z-50 animate-fade-in">
                <div className="px-4 py-3 border-b border-surface-border">
                  <h3 className="text-sm font-semibold text-text-primary">Settings</h3>
                </div>
                <div className="p-4 space-y-4">
                  {/* Model */}
                  <div>
                    <label className="block text-xs font-medium text-text-secondary mb-1">Model</label>
                    <select
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-lg text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent disabled:opacity-50 transition-colors"
                    >
                      <optgroup label="Anthropic">
                        <option value="aws/anthropic/claude-opus-4-5">Claude Opus 4.5</option>
                        <option value="aws/anthropic/claude-haiku-4-5-v1">Claude Haiku 4.5</option>
                      </optgroup>
                      <optgroup label="OpenAI">
                        <option value="azure/openai/gpt-5.2">GPT 5.2</option>
                        <option value="azure/openai/gpt-5.1">GPT 5.1</option>
                        <option value="azure/openai/gpt-5.1-codex">GPT 5.1 Codex</option>
                        <option value="azure/openai/o4-mini">O4 Mini</option>
                        <option value="azure/openai/o1">O1</option>
                      </optgroup>
                      <optgroup label="Google">
                        <option value="gcp/google/gemini-3-pro">Gemini 3 Pro</option>
                        <option value="gcp/google/gemini-3-pro-preview">Gemini 3 Pro Preview</option>
                        <option value="gcp/google/gemini-3-flash-preview">Gemini 3 Flash Preview</option>
                        <option value="gcp/google/gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
                      </optgroup>
                      <optgroup label="Open Source">
                        <option value="nvdev/deepseek-ai/deepseek-v3-0324">DeepSeek V3</option>
                        <option value="deepseek-ai/deepseek-r1-0528">DeepSeek R1</option>
                        <option value="nvdev/qwen/qwen-235b">Qwen 235B</option>
                        <option value="moonshotai/kimi-k2-instruct">Kimi K2</option>
                      </optgroup>
                    </select>
                  </div>

                  {/* Server URL */}
                  <div>
                    <label className="block text-xs font-medium text-text-secondary mb-1">Server URL</label>
                    <input
                      type="text"
                      value={serverUrl}
                      onChange={(e) => setServerUrl(e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-lg text-sm text-text-primary placeholder-text-tertiary focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent disabled:opacity-50 transition-colors"
                    />
                  </div>

                  {/* Temperature */}
                  <div>
                    <label className="block text-xs font-medium text-text-secondary mb-1">Temperature</label>
                    <input
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
                      disabled={isRunning}
                      min="0"
                      max="2"
                      step="0.1"
                      className="w-24 px-3 py-2 bg-surface-sunken border border-surface-border rounded-lg text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/30 focus:border-accent disabled:opacity-50 transition-colors"
                    />
                  </div>

                  {/* Pause each turn */}
                  <label className="flex items-center gap-2.5 cursor-pointer group">
                    <input
                      type="checkbox"
                      checked={awaitUserInput}
                      onChange={(e) => {
                        const newValue = e.target.checked;
                        setAwaitUserInput(newValue);
                        if (isRunning) {
                          trial.updateSettings({ await_user_input_each_turn: newValue });
                        }
                      }}
                      className="w-4 h-4 rounded border-surface-border-light text-accent focus:ring-accent/30 focus:ring-offset-0"
                    />
                    <span className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">
                      Pause each turn for feedback
                    </span>
                  </label>
                </div>
              </div>
            )}
          </div>
          </div>
        </div>
      </header>

      {/* Main Content — Resizable Split */}
      <main ref={containerRef} className="flex-1 flex overflow-hidden relative">
        {/* Overlay to capture mouse during drag (prevents iframe from stealing events) */}
        {isDragging && <div className="absolute inset-0 z-20" />}

        {/* Left Panel - Chat */}
        <div
          className="flex flex-col bg-surface overflow-hidden"
          style={{ width: `${splitPercent}%` }}
        >
          <ChatPanel
            messages={trial.messages}
            state={trial.state}
            onSendMessage={trial.injectPrompt}
            onResume={trial.resumeTrial}
            taskPrompt={trial.taskPrompt}
          />
        </div>

        {/* Draggable Divider */}
        <div
          className={`flex-shrink-0 w-1.5 cursor-col-resize relative group z-30 transition-colors ${
            isDragging ? 'bg-accent' : 'bg-surface-border hover:bg-accent/60'
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Wider invisible hit target */}
          <div className="absolute inset-y-0 -left-1.5 -right-1.5" />
          {/* Visual grip dots */}
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="w-1 h-1 rounded-full bg-sand-400" />
            <div className="w-1 h-1 rounded-full bg-sand-400" />
            <div className="w-1 h-1 rounded-full bg-sand-400" />
          </div>
        </div>

        {/* Right Panel - Visualization */}
        <div
          className="flex flex-col bg-surface-raised overflow-hidden"
          style={{ width: `${100 - splitPercent}%` }}
        >
          <VisualizationPanel />
        </div>
      </main>
    </div>
  );
}

export default App;
