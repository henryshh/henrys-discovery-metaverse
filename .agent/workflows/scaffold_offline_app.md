---
description: Initialize and convert a React application into a single offline HTML file.
---

# Scaffold Offline App Workflow

This workflow describes the process of converting a React-based HTML project into a fully offline, self-contained application. It incorporates robust handling for common issues like regex replacement bugs, React version mismatches, and console warnings.

## Prerequisites

-   Node.js installed.
-   Access to library files (e.g., `c:/Libs-all`) or a way to download them.

## Steps

### 1. Analyze Dependencies

Identify all `<script src="...">` tags in your source HTML. Determine which are:
-   **ES Modules (ESM)**: React, ReactDOM (need import maps).
-   **Classic Scripts**: Babel, Tailwind, Date-fns (need direct inlining).

### 2. Create Conversion Script (`convert_offline.js`)

Create a `convert_offline.js` script in your project root. Use the following robust patterns:

#### A. Robust Regex Replacement
**Problem:** `string.replace(regex, string)` fails if the replacement string contains special patterns like `$'` (common in minified code).
**Solution:** Always use a callback function for replacement.
```javascript
htmlContent.replace(regex, () => replacementString);
```

#### B. Safe Script Tag Escaping
**Problem:** Inlining code containing `</script>` breaks the HMTL parser.
**Solution:** Hex-escape the closing tag.
```javascript
scriptContent.replace(/<\/script/gi, '\\x3c/script');
scriptContent.replace(/<\\\/script/gi, '\\x3c\\/script'); // Handle existing escapes
```

#### C. React & Scheduler Shim
**Problem:** `react-dom` (v19+) imports `scheduler`, which isn't export by the default `react` bundle. `createRoot` is in `react-dom/client`.
**Solution:**
1.  Bundle `scheduler.js` separately.
2.  Add `scheduler` to your Import Map.
3.  Patch `react-dom` imports to point to `scheduler`.
4.  Merge `react-dom` and `react-dom/client` exports in the shim.

```javascript
// Import Map Template
const ESM_SHIM = `
<script type="importmap">
{
  "imports": {
    "react": "data:text/javascript;base64,REACT_B64",
    "react-dom": "data:text/javascript;base64,REACT_DOM_B64",
    "react-dom/client": "data:text/javascript;base64,REACT_DOM_CLIENT_B64",
    "scheduler": "data:text/javascript;base64,SCHEDULER_B64"
  }
}
</script>
`;
```

#### D. Console Warning Suppression
**Problem:** Libraries like Tailwind and Babel warn about production usage.
**Solution:** Patch the source code during inlining to replace `console.warn(...)` with `void(0)`.

```javascript
if (lib.filename === 'tailwind.js') {
    content = content.replace(/console\.warn\(\s*['"]cdn\.tailwindcss\.com should not be used in production[^'"]*['"]\)/, '({/* suppressed tailwind warning */})');
}
```

### 3. Execute Conversion

Run the script to generate your offline file:
```bash
node convert_offline.js
```

### 4. Verification

-   **Open in Browser**: Check for `SyntaxError` (escaping issues) or `TypeError` (import issues).
-   **Check Console**: Ensure it is clean of warnings.
-   **Functionality**: Test interactive features (React components, date parsing).

---

## Troubleshooting

-   **"Ol is not a function"**: Missing `scheduler` import or incorrect mapping.
-   **"ReactDOM.createRoot is not a function"**: Missing `react-dom/client` mapping or shim merge logic.
-   **Duplicated Code / massive file size**: Regex replacement bug (see Step 2A).

## Key Skills & Patterns

To avoid these errors in future offline conversions, apply these core skills:

1.  **Code-as-Data Safety**: Treat source code as "unsafe data" when manipulating it. Never use simple string replacement for code injection. Always use callbacks or AST-based tools.
2.  **Explicit Dependency Modeling**: Verification of implicit dependencies (like `react-dom` -> `scheduler`) is critical when moving from a bundler (webpack/vite) to manual module management.
3.  **Environment Adaptation**: Anticipate environment differences (Browser vs Node vs Offline File). Stub out browser-specific APIs (`console.warn`, `window.onerror`) that don't make sense in the target environment.
4.  **Hex Escaping**: The "Double-Escape" pattern (`\\x3c` for `<`) is the only robust way to inline HTML/JS into a single file without breaking the parser.

## "Golden Rules" for Offline Mode (from OFFLINE_GUIDE_ZH)

1.  **Dual Mode Development**:
    *   **Dev Mode**: Standard multi-file setup with `file://` or `liveserver`.
    *   **Build Mode**: Script (`convert_offline.js`) that inlines everything. Never code directly in the single file.

2.  **Zero External Requests**:
    *   **No CDNs**: Reject `esm.sh`, `unpkg`, `cdnjs`. Download full UMD/ESM bundles locally.
    *   **Fonts**: Use system fonts (`sans-serif`, `system-ui`) or inline Base64 woff2. No Google Fonts links.
    *   **Icons**: Use SVG components (JS/React), not font files.

3.  **Dependency Rigor**:
    *   **Validation**: Inspect every `.js` file. If it contains `import ... from "https://..."`, it is a Shim, NOT a Bundle. **Reject it.**
    *   **Patching**: You MUST strictly patch internal imports (e.g., inside `react-dom`, replace `from "/react"` with `from "react"` to match your Import Map).

4.  **On-Screen Debugging**:
    *   Inject a global `window.onerror` handler at the very top of `<head>` to catch white-screen errors (Import Map failures) and display them with a red overlay. Users cannot debug white screens without this.
