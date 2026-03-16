# Frontend Changes

## Dark/Light Mode Toggle Button

### Summary
Added a dark/light mode toggle button positioned in the top-right corner of the UI.

### Files Modified

#### `frontend/index.html`
- Added a `<button id="themeToggle">` element with `position: fixed` (via CSS), placed before `.container`
- Includes two inline SVGs: a sun icon (`icon-sun`) and a moon icon (`icon-moon`)
- Button has `aria-label="Toggle dark/light mode"` and `title="Toggle theme"` for accessibility
- Updated stylesheet version query param to `?v=11` to bust cache

#### `frontend/style.css`
- Added a `body[data-theme="light"]` block with light theme CSS variable overrides:
  - Background: `#f8fafc`, Surface: `#ffffff`
  - Text: `#0f172a` / `#64748b`
  - Borders: `#e2e8f0`
  - Welcome message uses soft blue tones
- Added two new CSS variables (`--toggle-bg`, `--toggle-hover-bg`) for the button background in each theme
- Added `.theme-toggle` styles:
  - `position: fixed; top: 1rem; right: 1rem; z-index: 100`
  - 40×40px circle shape matching existing button aesthetic
  - Hover: scale(1.1) + box-shadow; Active: scale(0.95)
  - Focus ring using `--focus-ring` variable (accessible keyboard nav)
  - Smooth transitions on background, border, color, transform
- Icon visibility rules:
  - Dark mode (default): shows sun icon (click = switch to light)
  - Light mode: shows moon icon (click = switch to dark)
- Added `transition: background-color 0.3s ease, color 0.3s ease` to `body`, `.sidebar`, `.message.assistant .message-content`, and `#chatInput` for smooth theme switching animation

#### `frontend/script.js`
- Added `initTheme()` — reads saved theme from `localStorage` and applies it on page load (before DOMContentLoaded to avoid flash)
- Added `applyTheme(theme)` — sets/removes `data-theme="light"` on `<body>`
- Added `toggleTheme()` — toggles between dark/light and persists choice to `localStorage`
- In `DOMContentLoaded`: wires up `click` listener on `#themeToggle` to call `toggleTheme()`

### Design Decisions
- Theme is stored in `localStorage` under the key `"theme"` (defaults to `"dark"`)
- `initTheme()` is called immediately (not inside DOMContentLoaded) to minimize flash of wrong theme
- Both icons are always in the DOM; CSS controls which is visible — avoids JS DOM manipulation on every toggle
- Uses existing CSS variable system, so all components that reference `--background`, `--surface`, etc. automatically adapt to the light theme without per-component overrides
