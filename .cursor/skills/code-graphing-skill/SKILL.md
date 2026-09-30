---
name: code-graphing-skill
description: Design and render diagrams for code, architecture, logic, data flow, control flow, state, interactions, and implementation plans. Use Mermaid for simple-to-moderate diagrams that should remain compact and Markdown-native; escalate to D2 when the view needs denser composition, richer styling, interactive SVG behavior, multiple boards, or animation. Writes diagram bundles (source, SVG, PNG) into the agent workspace and validates renders before delivery.
compatibility: Requires POSIX shell (Git Bash on Windows), curl, Node.js >=18, and npm for first-time setup. Renderer dependencies install into .diagram-tools inside this skill and are reused thereafter.
metadata:
  mermaid-cli: "11.17.0"
  d2: "0.9.0"
---

# Code Graphing

Turn a code or logic question into the smallest diagram that makes the relevant structure obvious. The diagram is an explanatory model, not a repository dump.

## Skill root vs workspace (read first)

| Location | What it is | What goes here |
|----------|------------|----------------|
| **Skill root** | Directory containing this `SKILL.md` (under your Pi agent install, e.g. `~/.pi/agent/skills/.../code-graphing-skill/`) | Skill instructions, `scripts/`, pinned `.diagram-tools/`, and bundled **examples** for smoke-testing the render scripts only |
| **Workspace** | The repository or folder the user’s task is about—the agent’s current working project | **All deliverable diagram bundles** for that task |

Never treat skill root or `~/.pi/` as the place to leave user-facing diagram outputs. Temporary renders during iteration may use a temp path, but **final artifacts must be copied into the workspace** before the task is complete.

Run `scripts/*.sh` from **skill root** in a POSIX shell. Pass **absolute or workspace-relative paths** for all inputs and outputs.

## 1. Establish the diagram contract

Before writing syntax, identify four things:

1. **Question** — the single question the diagram must answer.
2. **Audience** — reader knowledge and the abstraction level they need.
3. **Scope** — which systems, modules, functions, states, steps, or data objects belong in view.
4. **Evidence** — which relationships are verified from code/input and which are inferred. Never invent implementation details to make the picture complete.

If the source is a codebase, inspect only enough code to establish the nodes, boundaries, ordering, and relationships required by the question. Preserve uncertainty in labels or notes rather than presenting guesses as facts.

## 2. Convert the subject into graph primitives

Model the content before choosing a renderer.

- **Node** — one meaningful entity: actor, service, module, function, state, datastore, event, decision, artifact, or step.
- **Edge** — one meaningful relationship or transition. Label it when the relationship is not obvious from direction alone.
- **Container** — a real boundary: subsystem, package, process, trust boundary, layer, phase, or ownership domain.
- **Order** — sequence or causality. Encode only when order matters.
- **Branch** — a decision, condition, alternative, error path, or state transition.
- **Multiplicity** — one-to-many, fan-in, fan-out, broadcast, aggregation, retry, or loop.
- **Annotation** — secondary information that helps interpretation but is not part of the graph itself.

Use one visual meaning consistently. Direction should have one interpretation within a diagram—normally control/request flow, dependency direction, data movement, or time.

## 3. Choose the diagram form

Select the form that matches the question rather than the source material:

- **Flow/process** — ordered logic, branching, validation, algorithms, CI/CD, user journeys.
- **Sequence** — time-ordered interactions among stable participants.
- **Architecture/dependency** — components, ownership boundaries, calls, imports, queues, storage, external systems.
- **State** — lifecycle, finite-state behavior, allowed transitions.
- **Class/ER/data model** — structural relationships, schemas, cardinality, type composition.
- **Decision tree** — choices and consequences where branch conditions are the main content.
- **Layered/multi-board explanation** — successive states, alternatives, progressive build-up, or changes over time.

Do not combine forms merely to show more information. Split into multiple views when one view would need multiple conflicting visual grammars.

## 4. Control complexity before rendering

Prefer omission and grouping over shrinking text.

- Keep each node semantically singular.
- Collapse repeated low-value internals into a container or summary node.
- Exclude unrelated dependencies, tests, generated files, and utility code unless they answer the question.
- Replace edge crossings with grouping or a different orientation before adding styling.
- Use short labels. Put detail in annotations, tooltips, notes, or accompanying prose.
- For code-derived diagrams, keep stable identifiers internally even when display labels are shortened.

Split the diagram when a reader must mentally track more than one abstraction level at once.

## 5. Select the renderer tier

### Tier 1 — Mermaid

Use Mermaid when all of the following are true or desirable:

- the diagram is primarily structural or sequential;
- Markdown-native source is valuable;
- standard flowchart, sequence, state, class, ER, architecture, timeline, or similar grammar is sufficient;
- one static board communicates the idea;
- styling and precise composition are secondary to portability.

Read **[references/mermaid.md](references/mermaid.md)** before authoring or rendering Mermaid.

### Tier 2 — D2

Use D2 when any material requirement exceeds the Mermaid tier:

- dense nested architecture or containers;
- stronger layout control or a need to try alternate layout engines;
- rich code/Markdown labels, SQL/UML structures, or extensive reusable styling;
- links or tooltips in an interactive SVG;
- multiple boards representing layers, scenarios, or successive steps;
- animated SVG/GIF output;
- live browser preview while iterating;
- the diagram remains valid but Mermaid layout is visually ambiguous after one restructuring attempt.

Read **[references/d2.md](references/d2.md)** before authoring or rendering D2.

Do not escalate merely for aesthetics. Escalate because the representation or output behavior requires it.

## 6. Prepare the standard runtime

Renderer dependencies live only at:

```text
<skill-root>/.diagram-tools/
```

Never install renderer packages into the workspace and never depend on an unpinned global renderer.

From skill root:

```bash
scripts/setup.sh
```

The script is idempotent. See renderer references for dependencies.

## 7. Default visual style

Bundled render scripts apply a **dark, modern** look by default:

- **Mermaid** — `assets/mermaid-theme.json` (dark palette, system UI font stack) plus canvas `#1e1e2e` via `scripts/render-mermaid.sh`.
- **D2** — `--theme=200` via `scripts/render-d2.sh` (dark theme; D2’s default font is Source Sans Pro).

Do not restyle per diagram unless the user asks. Optional overrides: pass extra renderer flags after the output path (see references).

## 8. Workspace output contract (mandatory)

This is the **only** deliverable format. Do not leave final diagrams only in skill root, `/tmp`, or Pi paths.

### When to use multiple bundles

- **Multiple charts** — different questions, explicit user request for several images, or a multi-part flow that does not fit one view → **one bundle folder per chart**.
- **One chart** — one bundle folder, even if the D2 source uses multiple boards (`steps` / `layers` / `scenarios`).

### Directory layout

Default parent directory in the **workspace** (unless the user names another):

```text
diagrams/
```

Each chart gets its own folder named with a short **kebab-case slug** describing the chart (not the renderer):

```text
<workspace>/diagrams/<chart-slug>/
  <chart-slug>.mmd   OR   <chart-slug>.d2    # source (exactly one)
  <chart-slug>.svg                          # canonical vector render
  <chart-slug>.png                          # canonical raster render
```

The three files above are a **bundle**. Every delivered chart must have all three unless the user explicitly waives a format (if waived, say so in the reply).

### Authoring and rendering workflow

1. Create `<workspace>/diagrams/<chart-slug>/`.
2. Write source as `<chart-slug>.mmd` or `<chart-slug>.d2` in that folder.
3. Render from skill root, pointing at workspace paths, for example:

```bash
scripts/render-mermaid.sh "<workspace>/diagrams/<chart-slug>/<chart-slug>.mmd" \
  "<workspace>/diagrams/<chart-slug>/<chart-slug>.svg"
scripts/render-mermaid.sh "<workspace>/diagrams/<chart-slug>/<chart-slug>.mmd" \
  "<workspace>/diagrams/<chart-slug>/<chart-slug>.png"
```

```bash
scripts/render-d2.sh "<workspace>/diagrams/<chart-slug>/<chart-slug>.d2" \
  "<workspace>/diagrams/<chart-slug>/<chart-slug>.svg"
scripts/render-d2.sh "<workspace>/diagrams/<chart-slug>/<chart-slug>.d2" \
  "<workspace>/diagrams/<chart-slug>/<chart-slug>.png"
```

4. **D2 multi-board** — if the renderer emits a subdirectory (e.g. `index.svg` inside `<chart-slug>/` because of `steps`), normalize to the bundle contract:
   - Copy or move `index.svg` → `<chart-slug>.svg` and `index.png` → `<chart-slug>.png`.
   - Keep additional board files as `<chart-slug>/<board-name>.svg` and `.png` alongside the bundle triplet when they add meaning; remove stray empty directories.
5. Inspect SVG/PNG; fix source and re-render until readable.
6. Reply to the user with **workspace-relative paths** to each bundle folder and the three files. That path list is the primary output; add a short description of what each chart shows and any material uncertainty.

Do not delete source after rendering. Do not deliver diagrams without the workspace bundle files.

## 9. Validation examples (skill root only)

Smoke-test render scripts after setup; outputs stay under skill `examples/` and are not user deliverables:

| Path | Tier | Notes |
|------|------|-------|
| [examples/simple-flow.mmd](examples/simple-flow.mmd) | Mermaid | Single-board flowchart |
| [examples/simple-arch.d2](examples/simple-arch.d2) | D2 | Single-board architecture |
| [examples/complex-flow.d2](examples/complex-flow.d2) | D2 | Multi-board `steps`; see [references/d2.md](references/d2.md) |
