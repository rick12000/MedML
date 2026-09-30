---
name: obsidian-orienteering
description: >-
  Vault placement for new notes and relocations via bounded folder search.
  Use when creating a note or moving/relocating a note or file in Obsidian
  and the destination folder is not fully specified. Pair with
  local-path-lookup and obsidian-syntax for vault path and markdown.
---

# Obsidian orienteering

Resolve **where** a vault note or file belongs before creating or moving it. Read `local-path-lookup` for the vault root. Ignore `.obsidian/` and other non-content roots unless the user targets them.

Apply this skill when:

- Creating a new note (or note-like file) without a fixed folder path.
- Relocating an existing note or file within the vault.

Skip when the user gives an explicit destination path and does not ask for placement review.

List directories with tools; do not invent vault structure. Stop at [User checkpoint](#user-checkpoint) until the user confirms when a checkpoint fires.

## Inputs

Collect before search:

- Operation: **create** or **relocate** (source path required for relocate).
- Artifact type: note, folder, Kanban (board file per vault convention), or other vault file.
- Working title or topic summary from the user prompt (for create); note title and content scope (for relocate).
- Optional: batch of planned files (changes folder rules below).

## Relevance model

Score each candidate directory against the artifact using **scope and purpose**, not lexical overlap of note bodies.

- **Scope:** what domain the directory is meant to hold (e.g. “programming languages”, “Obsidian workflow”, “experiment design”).
- **Purpose:** how notes in that directory are used together (reference, tutorial, style guide, project log).
- Treat cross-language or cross-format pairs as similar when they share scope (e.g. Python package docs and C++ tutorial → both “language technical reference”; Obsidian syntax reference and Obsidian prose style → both “Obsidian authoring”).

Assign each candidate a discrete judgment: **pursue** (clear fit), **weak** (plausible but uncertain), **reject** (no meaningful fit).

Never advance into **reject** candidates solely because they rank highest among bad options.

## Folder balance (apply during scoring and proposals)

Target a middle depth: specialized enough to navigate, not so narrow that folders become single-file graves.

| Signal | Interpretation | Bias |
|--------|----------------|------|
| Many siblings at one level (wide, shallow) | Under-specialized | Prefer descending or proposing subfolders when file count is high |
| Directory with 1–2 notes and no growth path | Over-specialized | Prefer parent or merge with scope-similar loose notes |
| Directory with “too many” loose notes (vault-relative judgment; typically dozens) | Under-specialized at that node | Prefer subfolders or scope-based splits |

Prefer **deeper, contained** trees over many top-level buckets. Do not create a new folder for one note unless an exception below applies.

## Search algorithm

State variables:

- `R`: current **round** (starts at 1).
- `C`: set of **candidate directory paths** under the vault (POSIX-style `/` segments relative to vault root).
- `L`: **last sensible directory** — deepest path where at least one **pursue** or **weak** candidate still justified continued search.

### Initialization

1. Set `C` ← immediate child directories of vault root (depth 1 only). Do not list files yet except where noted in checkpoints.
2. Set `L` ← vault root.

### Round loop

For each round while `C` is non-empty:

1. **Score** every path in `C` (full path labels required whenever `R > 1`; round 1 uses single segment names).
2. **Prune:** remove **reject** paths from consideration for descent.
3. **Stop check:** if no remaining path is **pursue**, or all remaining are only **weak** and the artifact is novel to the vault (topic not reflected anywhere in tree), **exit loop** → [User checkpoint](#user-checkpoint). Set `L` to the deepest directory that still had a **pursue** parent chain; if round 1 had zero **pursue**, `L` is vault root.
4. **Select** up to **three** highest-value **pursue** paths (tie-break: shallower path, then fewer sibling files if known). If fewer than three **pursue**, include **weak** only to fill exploration slots, never more than three total.
5. Update `L` to the best **pursue** path in this round (or best **weak** if no **pursue** but stop check did not fire).
6. **Expand:** for each selected path `P`, list immediate child directories only. Form child labels as `P/childName` (full composite path from vault root).
7. Union all expanded children into set `N`.
8. If `N` is empty: **leaf** — place artifact at `L` (or at the single selected leaf `P` if exactly one **pursue** path was selected and it has no children). **End search.**
9. Set `C` ← `N`, increment `R`, continue loop.

### After leaf resolution

Before `write` / `mkdir` / move:

- Count existing notes in target directory (non-recursive unless user asked for nested inventory).
- If count triggers balance rules, adjust destination or add a subfolder proposal in the checkpoint instead of silently overfilling.

## User checkpoint

Halt search when relevance fails, uncertainty is high, or balance rules conflict with automatic placement. Present:

1. **Findings:** vault root, rounds explored, which paths were **pursue** / **weak** / **reject**, and current `L`.
2. **Loose files at `L`:** list `.md` files directly in `L` (not subfolders) when considering grouping.

Offer exactly one primary recommendation and label alternatives:

| ID | Proposition | When |
|----|-------------|------|
| **A** | Place artifact in `L` (not in a child) | Descent no longer justified but parent scope still fits; or no depth-1 folder was **pursue** (use vault root as `L`). |
| **B** | Create new folder under `L` and place artifact plus existing loose notes | ≥2 files at `L` (including the new artifact) share scope/purpose; name folder after that shared scope. |
| **C** | Create new folder for this artifact alone under `L` | User confirms future related notes **or** the task already includes multiple files to create in one batch. |

Require explicit user confirmation before **B** or **C**. Default suggestion is **A** unless **B** or **C** criteria clearly apply.

## Relocate

Run the same search with operation **relocate**; existing note content informs scope. Destination leaf replaces “create”. Perform the filesystem move only after path confirmation (checkpoint if needed).

## Output

Return confirmed absolute path under the vault, the proposition ID accepted (if any), and a one-line rationale tied to scope/purpose (not keyword matching).

## Post-relocate link hook

After any **relocate**, scan the vault for links that still point at the old location or stale path-qualified name. Title-only wikilinks (`[[Note Title]]`) often survive a folder move; path-qualified wikilinks, embeds, and markdown path links frequently do not.

1. Record **old** relative path from vault root (directory + basename) and **new** relative path after the move. If the filename changed, record old and new basenames.
2. Search all vault `.md` (and other text the user treats as notes) for:
   - Wikilinks and embeds: `[[...]]`, `![[...]]` containing the old path prefix, old folder segments, or old basename when disambiguation was path-based.
   - Markdown links: `](...)` with relative paths into the old directory.
   - Transclusion blocks or plugin syntax that embed paths if present in the vault.
3. Update each hit to the correct target: path-qualified link → new `folder/Note` form per vault convention; relative markdown links → recompute from the linking file’s new relationship to the target.
4. Re-scan for the old path string and old path-qualified link forms; fix stragglers.
5. Report files changed and link patterns updated. If ambiguous duplicate note titles exist, list conflicts and ask the user before rewriting.

Run this hook before considering the relocation task complete.
