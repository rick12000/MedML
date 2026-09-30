# Mermaid Tier

Use this reference after the parent skill selects Mermaid.

## 1. Runtime contract

Pinned renderer: `@mermaid-js/mermaid-cli@11.17.0`.

Required host tools:

- Node.js >=18
- npm
- POSIX shell

The local package and its Puppeteer/browser dependencies are installed only under:

```text
<skill-root>/.diagram-tools/mermaid/
```

Initialize or verify with (from skill root, POSIX shell):

```bash
scripts/setup.sh mermaid
```

Do not use a global `mmdc` when the bundled runtime is available. The pinned local binary is:

```text
.diagram-tools/mermaid/node_modules/.bin/mmdc
```

Mermaid CLI renders `.mmd` to SVG, PNG, or PDF. Its raster/vector rendering uses a headless browser. On minimal Linux images, Chromium may require OS shared libraries; treat those as host prerequisites and report the missing library instead of silently modifying the operating system.

## 2. Choose the Mermaid grammar

Use the narrowest grammar that directly represents the concept.

### Flowchart

Use for algorithms, request paths, control flow, decisions, transformations, deployment flows, and compact architecture sketches.

- `TD` / `TB`: hierarchy, pipelines, top-down processes.
- `LR`: request/data flow, dependency chains, architecture.
- Decision nodes represent actual conditions, not arbitrary emphasis.
- Subgraphs represent real boundaries, not decorative boxes.

### Sequence diagram

Use when time and participant interaction are primary.

- Participants are stable actors/components.
- Messages state meaningful operations, events, or payload intent.
- Use `alt`, `opt`, `loop`, `par`, and notes only for behavior that changes interpretation.
- Include failure/timeout branches when they are part of the question.

### State diagram

Use when the legal states and transitions are primary. Label transitions with the triggering event or condition. Do not represent implementation call order as states.

### Class diagram

Use for type relationships and public structure. Show only fields/methods relevant to the question; omit implementation trivia.

### ER diagram

Use for entities, relationships, and cardinality. Include key fields only when needed to understand joins or ownership.

### Architecture diagram

Use `architecture-beta` for cloud/service/resource topology when its groups/services/edges map naturally to the system. Prefer ordinary flowcharts when the architecture syntax would force an unnatural model.

## 3. Authoring rules

1. Start with semantics only: nodes, relationships, groups, ordering.
2. Use stable short IDs and readable labels.
3. Quote/escape labels that contain syntax-sensitive punctuation.
4. Keep edge text short: verb, protocol, event, or data category.
5. Avoid per-node styling until the graph reads correctly; bundled renders use the shared dark theme (see §5).
6. When styling is necessary, use a small semantic class vocabulary via `classDef`; do not style every node independently.
7. Prefer one direction. If subgraphs need different internal directions, verify the rendered layout because Mermaid may constrain them based on external connections.
8. Avoid relying on manual coordinates; Mermaid is an automatic-layout renderer.

## 4. Complexity limits and escalation

Restructure once before escalating to D2.

Escalate when the intended semantics require any of these:

- multiple simultaneously meaningful nested boards;
- interactive links/tooltips as part of the deliverable;
- animation or progressive state changes;
- rich embedded Markdown/code content;
- layout behavior still obscures the graph after grouping/orientation changes;
- extensive styling begins to dominate source readability.

Do not escalate solely because the diagram has many nodes. First reduce scope.

## 5. Rendering

Render through the bundled wrapper from **skill root**. Output paths must live in the **workspace** bundle folders defined in the parent skill (not under skill root).

```bash
scripts/render-mermaid.sh "<workspace>/diagrams/<chart-slug>/<chart-slug>.mmd" \
  "<workspace>/diagrams/<chart-slug>/<chart-slug>.svg"
scripts/render-mermaid.sh "<workspace>/diagrams/<chart-slug>/<chart-slug>.mmd" \
  "<workspace>/diagrams/<chart-slug>/<chart-slug>.png"
```

`scripts/render-mermaid.sh` applies `assets/mermaid-theme.json` and canvas `#1e1e2e` by default. Optional CLI arguments after the output path are forwarded to `mmdc`, for example:

```bash
scripts/render-mermaid.sh diagram.mmd diagram.png -s 2
```

Always produce **both** SVG and PNG for each bundle unless the user explicitly waives one format.

## 6. Validation loop

The renderer invocation is the syntax validation step: a non-zero exit means the source is not deliverable.

After a successful render, visually inspect the SVG or PNG and verify:

- no cropped text or shapes;
- no unintended overlaps;
- labels are readable at normal viewing size;
- edge directions match the intended semantics;
- decision branches are distinguishable;
- subgraphs correspond to real boundaries;
- important paths are visually obvious without relying on color alone.

If layout is poor, fix in this order:

1. remove irrelevant nodes/edges;
2. shorten labels;
3. change orientation;
4. regroup nodes/subgraphs;
5. split into multiple diagrams;
6. only then add limited styling or escalate to D2.

## 7. Output contract

Follow the parent skill **workspace output contract**: each chart is a folder under `<workspace>/diagrams/<chart-slug>/` containing `<chart-slug>.mmd`, `<chart-slug>.svg`, and `<chart-slug>.png`. Tell the user those workspace paths in the reply.

When embedding in Markdown elsewhere, a fenced `mermaid` block is optional; the bundle files in the workspace remain the canonical deliverable.
