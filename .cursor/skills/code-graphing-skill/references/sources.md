# Research provenance

This skill was synthesized rather than copied from one upstream skill. The external patterns retained were limited to high-value procedural ideas: question-first diagram design, renderer-specific progressive disclosure, local rendering, render-and-inspect validation, and explicit source-of-truth guardrails.

## Primary technical sources

- Agent Skills specification — directory layout, YAML frontmatter, references/scripts, relative paths, progressive disclosure, validation conventions: https://agentskills.io/specification
- Mermaid CLI — local installation, `mmdc`, SVG/PNG/PDF rendering, Node API, browser-based renderer: https://github.com/mermaid-js/mermaid-cli
- Mermaid documentation — diagram grammars and syntax: https://mermaid.js.org/
- D2 documentation — installation, language, exports, layouts, interactivity, watch mode, composition, animation: https://d2lang.com/tour/
- D2 releases — pinned D2 version: https://github.com/d2lang/d2/releases

## Public agent-skill patterns reviewed

- `mgranberry/mermaid-diagram-skill` and maintained fork `trikitrok/mod-mermaid-diagram-skill` — useful patterns: visual validation, evidence/source guardrails, theme separation, renderer-relative paths.
- `arjunprabhulal/agent-skills` diagramming skill — useful pattern: a diagram should answer one question; arrows should state meaningful transferred relationships.
- `magnus919/agent-skills` Mermaid skill — useful patterns: audience/job framing and renderer-aware validation.
- `jonmagic/skills` D2 skill — useful patterns: explicit D2 CLI workflow, layout switching, watch mode, SVG/PNG rendering.
- `chunhualiao/skill-mermaid-diagrams` — useful pattern: generate + validate + preserve `.mmd`, `.svg`, and `.png` artifacts.

These repositories are public examples, not security certifications. No external executable code from them is bundled in this skill.
