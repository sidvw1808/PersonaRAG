# PersonaRAG web app

React 19 + TypeScript + Vite client for the PersonaRAG FastAPI backend. See the [main README](../README.md) for the full project.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /api/* to the API on :8000
npm run build    # outputs dist/, which `uvicorn api:app` serves at /
npm run lint
```

- `src/api.ts`: typed client and NDJSON stream reader for `/chat/stream` and `/panel/stream`
- `src/components/Answer.tsx`: safe minimal markdown, with `[n]` citations rendered as buttons
- `src/components/SourcesDrawer.tsx`: the retrieved passages, with similarity and re-rank scores
- Set `VITE_API_URL` to point the app at an API on another origin.
