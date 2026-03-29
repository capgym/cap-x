# CaP-X Interactive Web UI

A modern chat-based interface for real-time robot code execution demos.

## Features

- **Chat Interface**: Interactive chat view showing model responses, code blocks, and execution results
- **Visual Feedback**: Display environment images captured during execution
- **Thinking Traces**: Collapsible sections for model reasoning
- **User Injection**: Inject custom prompts during multi-turn execution
- **3D Visualization**: Embedded iframe for live robot visualization (Viser)
- **Real-time Updates**: WebSocket streaming for instant feedback

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (React + Vite)                      │
│  ┌─────────────────────────────┬───────────────────────────────────┐│
│  │      Chat Panel (50%)       │    3D Visualization (50%)         ││
│  └─────────────────────────────┴───────────────────────────────────┘│
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ WebSocket
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (port 8200)                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.10+ with `uv` package manager
- Node.js 18+ with npm

### Backend Setup

The backend is part of the CaP-X package. No additional setup needed.

### Frontend Setup

```bash
cd web-ui
npm install
```

> **After editing frontend code** (e.g. `src/App.tsx`), rebuild the production bundle before launching:
> ```bash
> cd web-ui && npm run build
> ```
> The `--web-ui True` flag in `launch.py` serves the built assets from `web-ui/dist/`. Without rebuilding, your changes won't take effect.

## Running

### 1. Start the Backend Server

```bash
# From the CaP-X root directory
uv run python -m capx.web.server --port 8200
```

### 2. Start the Frontend Dev Server

```bash
cd web-ui
npm run dev
```

### 3. Start Viser Visualization (optional)

Make sure the 3D visualization server is running at http://localhost:8080/

### 4. Open the UI

Navigate to http://localhost:5173 in your browser.

## Usage

1. **Load Config**: Select a YAML config file from the dropdown (e.g., `env_configs/agibot/...`)
2. **Configure Model**: Select the model and adjust server URL if needed
3. **Start Trial**: Click "Start Trial" to begin execution
4. **Monitor**: Watch the chat for model responses, code execution, and visual feedback
5. **Interact**: When "Pause each turn" is enabled, inject prompts during execution
6. **Stop**: Click "Stop" to abort at any time

## Development

### Frontend Structure

```
src/
├── components/
│   ├── ChatPanel.tsx       # Main chat container
│   ├── ChatMessage.tsx     # Individual message rendering
│   ├── CodeBlock.tsx       # Syntax-highlighted code
│   ├── ThinkingSection.tsx # Collapsible reasoning
│   ├── ImageViewer.tsx     # Visual feedback display
│   └── ...
├── hooks/
│   ├── useWebSocket.ts     # WebSocket connection
│   └── useTrialState.ts    # Trial state machine
└── types/
    └── messages.ts         # TypeScript types
```

### Backend Structure

```
capx/web/
├── server.py              # FastAPI app + WebSocket
├── models.py              # Pydantic schemas
├── session_manager.py     # Session tracking
└── async_trial_runner.py  # Async trial execution
```

## WebSocket Protocol

### Server → Client Events

| Event | Description |
|-------|-------------|
| `model_thinking` | Model is generating |
| `model_response` | Model response with code |
| `code_execution_start` | Code block starting |
| `code_execution_result` | Execution completed |
| `visual_feedback` | Environment image |
| `user_prompt_request` | Awaiting user input |
| `trial_complete` | Trial finished |
| `state_update` | State changed |
| `error` | Error occurred |

### Client → Server Commands

| Command | Description |
|---------|-------------|
| `inject_prompt` | Add user text to prompt |
| `stop` | Stop the trial |
| `resume` | Continue without injection |
