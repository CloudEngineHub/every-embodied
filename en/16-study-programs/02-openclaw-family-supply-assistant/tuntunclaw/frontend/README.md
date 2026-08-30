# Frontend Instructions

This is the static web front end of OpenClaw.

## Files

- `index.html`
- `styles.css`
- `app.js`

## Usage

You can open `index.html` directly in a browser, or after starting a local FastAPI service through `main.py` in the root directory of the repository, you can access the web page.

The default state is the mock mode. When a real backend is to be connected later, the backend base address can be set through the `API` button at the top.
The front end is expected to call:

- `POST /api/command`
- `GET /api/session/:id`
- `GET /api/session/:id/events`

## Instructions

- The page uses a dark background and highlighted text.
- Command input supports `Enter` for submission and `Shift+Enter` for line breaks.
- Common test commands are pre-set for easy quick testing.
