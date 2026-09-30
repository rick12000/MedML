---
name: obsidian-syntax
description: >-
  Obsidian wiki links, embeds, callouts, math, and fenced code. Use when
  drafting, editing, or reviewing notes in Obsidian markdown. Skip for
  non-Obsidian markdown or general prose style.
---

# Obsidian syntax

Reference for Obsidian-flavored markdown only. Does not cover vault organization or prose voice.

## Wiki links

Obsidian resolves links by **note title** (filename without extension), not path. Paths are optional when disambiguation is needed.

| Syntax | Resolves to |
|--------|-------------|
| `[[Note Name]]` | Note whose filename is `Note Name.md` |
| `[[Note Name#Heading]]` | Block/heading within that note (heading text after `#`) |
| `[[#Heading]]` | Heading in the **current** note |
| `[[Note Name\|Display Text]]` | Link with custom anchor text (`\|` is the alias separator) |

- Heading anchors match the heading string Obsidian generates (case and punctuation matter).
- Prefer stable heading text; renaming headings breaks `#` links.

## Embeds (transclusion)

Leading `!` transcludes content instead of linking.

| Syntax | Behavior |
|--------|----------|
| `![[Note Name]]` | Embeds full note body at cursor |
| `![[Note Name#Heading]]` | Embeds section under that heading |
| `![[image-name.png]]` | Embeds image; default vault path is often `Attachments/` (depends on user settings) |

Embeds are block-level in the rendered note. Nested embeds follow Obsidian’s depth and cycle limits.

## Callouts

Blockquote-based admonitions. First line: type and optional title. Every content line must start with `> `.

**Basic:**

```markdown
> [!info] Title optional
> Line one.
> Line two.
```

**Collapsed by default:** append `-` immediately after the type token (no space): `[!info]-`

```markdown
> [!info]- Collapsed title
> Hidden until expanded.
```

**Types (built-in / common in vaults):**

| Token | Typical use |
|-------|-------------|
| `[!info]` / `[!info]-` | Neutral blocks, diagrams, long supplements |
| `[!note]` / `[!note]-` | Aside, qualification, proofs |
| `[!warning]` | Caveats, assumption breaks |
| `[!definition]` | Formal definitions (may require theme/plugin) |
| `[!example]` / `[!example]-` | Worked examples |
| `[!success]` | Results / interpretation |
| `[!question]` | Problem statements |
| `[!important]` | Hard constraints |

**Multi-line rules:**

- Prefix every line with `> `, including blanks inside the callout.
- For display math inside a callout, use blank `> ` lines around `$$ ... $$`:

```markdown
> [!info]- Derivation
> Setup:
>
> $$ E[\bar{X}] = \mu $$
>
> Therefore ...
```

**Nesting:** inner callout lines are still `> `-prefixed; one level of nesting is usually enough.

Prefer `[!type]-` over HTML `<details>` for collapsible bodies (renderer consistency).

## Mathematics

| Form | Delimiters | Rendering |
|------|------------|-----------|
| Inline | `$...$` | Flows inside a paragraph |
| Display | `$$` on its own lines | Centered block equation |

**Display block** — blank line before opening `$$`, equation lines, blank line after closing `$$`:

```markdown
Prior line of text.

$$
\mathrm{Var}(\bar{X}) = \frac{\sigma^2}{n}
$$

Next line of text.
```

- Literal `$` in prose: escape as `\$`.
- Long derivations: break lines at relation operators (`=`, `\leq`, etc.); keep one logical equation per `$$` block unless using aligned environments supported by the user’s math plugin.

**Inside callouts:** inline `$...$` works on `> ` lines; display math needs surrounding blank `> ` lines (see Callouts).

## Code

| Form | Syntax |
|------|--------|
| Inline | `` `identifier` `` or `` `command` `` |
| Fenced | Opening fence: `` ```lang `` then body then closing `` ``` `` |

- Always set a language id on fences (`python`, `bash`, `js`, `text`, etc.).
- Blank line before and after fenced blocks when adjacent to paragraphs or headings.

```python
def f(x):
    return x**2
```

## Filename constraints (link targets)

Wiki links use note filenames. On Windows, avoid characters invalid in paths (`\ / : * ? " < > |`). No requirement to encode spaces in `[[...]]`; match the note title exactly.
