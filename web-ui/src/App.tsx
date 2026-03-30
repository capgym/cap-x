import { useState, useEffect, useRef, useCallback } from 'react';
import { useTrialState } from './hooks/useTrialState';
import { ConfigStartControl } from './components/ConfigStartControl';
import { ChatPanel } from './components/ChatPanel';
import { VisualizationPanel } from './components/VisualizationPanel';

const FALLBACK_CONFIG = 'env_configs/cube_stack/franka_robosuite_cube_stack.yaml';

function App() {
  const trial = useTrialState();
  const [model, setModel] = useState('google/gemini-3.1-pro-preview');
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
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
          <span className="text-sm font-medium text-text-secondary tracking-wide">Initializing</span>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-surface">
      {/* Header */}
      <header className="flex-shrink-0 bg-surface-raised border-b border-surface-border header-glow">
        <div className="flex items-center h-16 px-6 gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2.5 flex-shrink-0">
            <img src="/capx_logo.svg" alt="CaP-X" className="w-7 h-7" />
            <h1 className="text-base font-bold text-text-primary tracking-[0.15em] uppercase">CaP-X</h1>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-surface-border flex-shrink-0" />

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

          {/* Right side: settings + status */}
          <div className="flex items-center gap-3 flex-shrink-0">
            {/* Status indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-surface-overlay border border-surface-border">
              <div className={`w-1.5 h-1.5 rounded-full ${
                trial.isConnected
                  ? isRunning
                    ? 'bg-accent animate-pulse'
                    : 'bg-nv-green'
                  : 'bg-text-tertiary'
              }`} />
              <span className="text-xs text-text-secondary font-medium tracking-wide">
                {isRunning ? 'Running' : trial.isConnected ? 'Ready' : 'Offline'}
              </span>
              {isRunning && <div className="w-12 h-0.5 rounded-full bg-accent/30 overflow-hidden"><div className="h-full w-1/2 bg-accent rounded-full animate-[slideIn_1s_ease-in-out_infinite_alternate]" /></div>}
            </div>

            {/* Settings Gear */}
            <div className="relative" ref={settingsRef}>
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2.5 rounded-md transition-all duration-150 ${
                showSettings
                  ? 'bg-surface-overlay text-accent border border-accent/20'
                  : 'text-text-tertiary hover:text-text-primary hover:bg-surface-overlay border border-transparent'
              }`}
              title="Settings"
              aria-label="Settings"
            >
              <svg className="w-4.5 h-4.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>

            {/* Settings Popover */}
            {showSettings && (
              <div className="absolute right-0 top-full mt-2 w-[calc(100vw-2rem)] sm:w-96 bg-surface-raised rounded-lg shadow-2xl border border-surface-border-light z-50 animate-scale-in">
                <div className="px-5 py-3.5 border-b border-surface-border flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-text-primary tracking-wide">Settings</h3>
                  <button
                    onClick={() => setShowSettings(false)}
                    className="p-2 text-text-tertiary hover:text-text-primary rounded transition-colors"
                    aria-label="Close settings"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <div className="p-5 space-y-5">
                  {/* Model */}
                  <div>
                    <label htmlFor="settings-model" className="block text-xs font-medium text-text-secondary mb-1.5 tracking-wide uppercase">Model</label>
                    <select
                      id="settings-model"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all cursor-pointer"
                    >
                      <optgroup label="Google">
                        <option value="google/gemini-3.1-pro-preview">Gemini 3.1 Pro Preview</option>
                        <option value="google/gemini-3.1-pro">Gemini 3.1 Pro</option>
                        <option value="google/gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
                      </optgroup>
                      <optgroup label="Anthropic">
                        <option value="anthropic/claude-opus-4-5">Claude Opus 4.5</option>
                        <option value="anthropic/claude-sonnet-4">Claude Sonnet 4</option>
                        <option value="anthropic/claude-haiku-4-5">Claude Haiku 4.5</option>
                      </optgroup>
                      <optgroup label="OpenAI">
                        <option value="openai/gpt-5.2">GPT 5.2</option>
                        <option value="openai/gpt-5.1">GPT 5.1</option>
                        <option value="openai/gpt-5.1-codex">GPT 5.1 Codex</option>
                        <option value="openai/o4-mini">O4 Mini</option>
                        <option value="openai/o1">O1</option>
                      </optgroup>
                      <optgroup label="Open Source">
                        <option value="deepseek/deepseek-v3-0324">DeepSeek V3</option>
                        <option value="deepseek/deepseek-r1-0528">DeepSeek R1</option>
                        <option value="qwen/qwen-235b-a22b">Qwen 235B</option>
                        <option value="moonshotai/kimi-k2-instruct">Kimi K2</option>
                      </optgroup>
                    </select>
                  </div>

                  {/* Server URL */}
                  <div>
                    <label htmlFor="settings-server-url" className="block text-xs font-medium text-text-secondary mb-1.5 tracking-wide uppercase">Server URL</label>
                    <input
                      id="settings-server-url"
                      type="text"
                      value={serverUrl}
                      onChange={(e) => setServerUrl(e.target.value)}
                      disabled={isRunning}
                      className="w-full px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary font-mono placeholder-text-tertiary focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all"
                    />
                  </div>

                  {/* Temperature */}
                  <div>
                    <label htmlFor="settings-temperature" className="block text-xs font-medium text-text-secondary mb-1.5 tracking-wide uppercase">Temperature</label>
                    <input
                      id="settings-temperature"
                      type="number"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value) || 0)}
                      disabled={isRunning}
                      min="0"
                      max="2"
                      step="0.1"
                      className="w-24 px-3 py-2 bg-surface-sunken border border-surface-border rounded-md text-sm text-text-primary font-mono focus:outline-none focus:ring-1 focus:ring-accent/40 focus:border-accent/40 disabled:opacity-40 transition-all"
                    />
                  </div>

                  {/* Pause each turn */}
                  <label className="flex items-center gap-2.5 cursor-pointer group py-1">
                    <div className="relative">
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
                        className="sr-only peer"
                      />
                      <div className="w-8 h-4.5 bg-surface-border rounded-full peer-checked:bg-accent/80 transition-colors" />
                      <div className="absolute top-0.5 left-0.5 w-3.5 h-3.5 bg-text-secondary rounded-full peer-checked:translate-x-3.5 peer-checked:bg-white transition-all" />
                    </div>
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
        {/* Overlay to capture mouse during drag */}
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
          className={`flex-shrink-0 w-0.5 cursor-col-resize relative group z-30 transition-colors duration-150 ${
            isDragging ? 'bg-accent gold-rule' : 'bg-surface-border hover:bg-accent/50'
          }`}
          onMouseDown={handleMouseDown}
        >
          {/* Wider invisible hit target */}
          <div className="absolute inset-y-0 -left-2 -right-2" />
          {/* Visual grip indicator */}
          <div className={`absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-3 h-8 rounded-full flex items-center justify-center transition-all ${
            isDragging ? 'bg-accent/20' : 'bg-transparent group-hover:bg-surface-overlay'
          }`}>
            <div className="flex flex-col gap-0.5">
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
              <div className="w-0.5 h-0.5 rounded-full bg-text-tertiary" />
            </div>
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
