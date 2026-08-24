# Design

## Source of truth

- Status: Active draft
- Last refreshed: 2026-08-24
- Primary product surfaces: model discovery, architecture exploration, resource estimation, hardware fit, and cost projection.
- Evidence reviewed: `src/pages/index.tsx`, `src/styles/globals.css`, `tailwind.config.mjs`, `src/components/model/*`, `README.md`, the supplied UI screenshots, and the canonical LABIIUM theme sources in `../labiium_web/src/app/globals.css`, `../labiium_web/src/lib/site-theme.ts`, and `../labiium_web/src/components/layout/ThemeCycleButton.tsx`.

## Brand

- Personality: LABIIUM photonic—technical, calm, investigative, trustworthy, with restrained optical depth rather than generic dashboard chrome.
- Trust signals: explicit provenance, immutable revisions, exact-versus-heuristic labels, source links, visible assumptions, and dated provider pricing tied to exact hardware.
- Avoid: generic “calculator” branding, unexplained AI hype, dense dashboard chrome, and unqualified accuracy claims.

## Product goals

- Goals: help users understand how a model is built, how memory scales, what hardware can host it, and where every estimate came from.
- Non-goals: executing arbitrary remote model code, replacing runtime benchmarks, or hiding uncertainty behind a single score.
- Success signals: users can move from a model ID to an understandable architecture flow and a defensible hardware/cost decision without leaving the app.

## Personas and jobs

- Primary personas: local model builders, ML engineers, infrastructure planners, and technically curious learners.
- User jobs: inspect a checkpoint, understand its attention/MLP/MoE structure, estimate KV-cache growth, compare hardware, and validate a deployment budget.
- Key contexts of use: desktop research, laptop/local-GPU planning, and responsive reference browsing.

## Information architecture

- Primary navigation: one explorer workspace with anchored sections: Inspect, Estimate, Understand, Hardware, Cost.
- Core routes/screens: the single static explorer route; progressive sections should remain usable independently.
- Content hierarchy: model identity/search first; a visual model → memory → hardware decision path second; interactive architecture third; tunable assumptions and detailed evidence on demand.

## Design principles

- Explain before optimizing: expose shapes, formulas, and provenance next to results.
- Progressive disclosure: keep the first view scannable; reveal source previews, detailed flows, and optional metadata on demand.
- Overview, focus, detail: lead with the complete deployment path, let users focus a graph component, then reveal implementation and formulas beside that focus instead of below a long canvas.
- Semantic zoom: zoom changes the information shown inside a component, not only its rendered size. Repeated layers are one bounded block with an external repetition count.
- Preserve page navigation: ordinary wheel/trackpad input scrolls the page; graph zoom requires an explicit control or Ctrl/⌘ modifier and uses small bounded steps.
- Evidence over decoration: visual emphasis should encode confidence, not merely importance.
- Tradeoffs: preserve a single-page static deployment while using compact summaries and disclosure controls to avoid a long-form dashboard.

## Visual language

- Color: use the LABIIUM Photon Palette exactly: photon black `#0b0e11`, photon white `#f5f7fa`, graphite surfaces, lab violet `#5333ed`, lab aqua `#04e2dc`, lab green `#12d88d`, lab amber `#ffb02e`, and lab sand `#efd8b5`.
- Themes: share LABIIUM's `System → Paper → Obsidian → Photonic` preference cycle and `labiium:theme` storage key so user preference follows across properties.
- Typography: Inter for UI/body, Plus Jakarta Sans 700–800 for display headings, and IBM Plex Mono 400/600 only for tensor shapes and identifiers.
- Spacing/layout rhythm: generous section spacing, compact control groups, max-width workspace, and a clear two-column results layout on desktop.
- Shape/radius/elevation: 14–20px glass cards in Photonic, flat hairline cards in Paper/Obsidian, restrained blur, and LABIIUM beam gradients for primary actions.
- Motion: short transitions for tabs, zoom, and selection; honor reduced-motion preferences.
- Imagery/iconography: simple line/icon glyphs and semantic diagrams; no decorative stock imagery.

## Components

- Existing components to reuse: DaisyUI controls, `Seo`, link/button primitives, `ModelEvidencePanel`, `ModelArchitectureDiagram`, `ArchitectureFlowExplorer`, and `KvCacheScalingCard`.
- New/changed components: deployment decision path, compact graph workspace with adjacent inspector, semantic graph nodes, and disclosure-based advanced controls.
- Variants and states: loading, error, stale query, source unavailable, heuristic, exact, fits, does-not-fit, selected flow node, zoomed flow.
- Token/component ownership: global theme stays in `globals.css`/DaisyUI; page-specific composition stays in `src/pages/index.tsx`; reusable model visuals stay under `src/components/model`.

## Accessibility

- Target standard: WCAG 2.1 AA intent.
- Keyboard/focus behavior: all tabs, zoom controls, flow nodes, source links, and form controls are keyboard reachable with visible focus; the graph cannot trap page scrolling or keyboard focus.
- Contrast/readability: do not use color as the only confidence signal; pair badges with text.
- Screen-reader semantics: use headings, labels, `aria-pressed` for view/node selection, and descriptive link text.
- Reduced motion and sensory considerations: keep transitions optional and avoid animated decoration.

## Responsive behavior

- Supported breakpoints/devices: mobile-first, with two-column estimation/results at large widths.
- Layout adaptations: stack cards and flow detail panels on narrow screens; keep diagrams horizontally scrollable rather than shrinking labels to illegibility.
- Touch/hover differences: controls remain tappable; hover is enhancement only.

## Interaction states

- Loading: show disabled fetch action and preserve the current estimate until replacement data arrives.
- Empty: explain how to fetch a public model and what evidence will appear.
- Error: identify gated/private/missing data and provide a safe next action.
- Success: show revision, sources, confidence, updated derived values, and cloud costs only when a rate matches the selected hardware.
- Disabled: preserve labels and explain why an action is unavailable.
- Offline/slow network: retain curated presets and catalog snapshots; avoid claiming live freshness.

## Content voice

- Tone: precise, direct, educational, non-alarmist.
- Terminology: “LLM Explorer,” “model evidence,” “architecture flow,” “KV cache,” “aggregate capacity,” and “reference snapshot.”
- Microcopy rules: distinguish exact arithmetic, sourced metadata, derived values, and heuristics in the label itself.

## Implementation constraints

- Framework/styling system: Next.js Pages Router, React, Tailwind CSS, DaisyUI, static export.
- Design-token constraints: no new design dependency; extend current theme and primitives.
- Performance constraints: no large client libraries for diagrams; bounded network inspection; preserve static build.
- Compatibility constraints: public browser inspection only; never execute arbitrary model code.
- Test/screenshot expectations: unit tests for derived model/visual state and safe zoom behavior, build/export verification, and visual verdict checks for UI iterations.

## Open questions

- [ ] Should future iterations split the single workspace into URL-addressable tabs, or keep one progressive explorer page?
- [ ] Should a future local analyzer emit a richer executable graph while the public browser view remains source-linked and read-only?
