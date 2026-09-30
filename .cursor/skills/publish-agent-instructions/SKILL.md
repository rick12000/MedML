---
name: publish-agent-instructions
description: Write one markdown instruction into the project files for Cursor, GitHub Copilot, Claude Code, Pi, Codex, Windsurf, and Cline. Use when publishing the same instructions so more than one agent can discover them.
---

# Publish agent instructions

Write the caller's body into every host below. Create parent directories. Do not change the body.

## Inputs

- `name`: file stem. Lowercase letters, numbers, and hyphens. This is also the Pi and Codex skill name.
- `description`: one line stating what the instruction does and when to use it.
- `body`: markdown with no frontmatter.

## Existing copy

An instruction named `<name>` already exists if any of these files exist at the repository root:

- `.cursor/rules/<name>.mdc`
- `.github/instructions/<name>.instructions.md`
- `.claude/rules/<name>.md`
- `.pi/skills/<name>/SKILL.md`
- `.agents/skills/<name>/SKILL.md`
- `.windsurf/rules/<name>.md`
- `.clinerules/<name>.md`

If the caller is reusing an existing instruction, stop and follow the file that exists. Do not write another copy.

## Files

Substitute `<name>`, `<description>`, and `<body>`.

`.cursor/rules/<name>.mdc`

```markdown
---
description: <description>
alwaysApply: true
---

<body>
```

`.github/instructions/<name>.instructions.md`

```markdown
---
applyTo: "**"
---

<body>
```

`.claude/rules/<name>.md`

```markdown
---
paths:
  - "**/*"
---

<body>
```

`.pi/skills/<name>/SKILL.md`

```markdown
---
name: <name>
description: <description>
---

<body>
```

`.agents/skills/<name>/SKILL.md`

Use the same contents as `.pi/skills/<name>/SKILL.md`. Pi discovers both. Keep the copies identical.

`.windsurf/rules/<name>.md`

```markdown
---
trigger: always_on
description: <description>
---

<body>
```

`.clinerules/<name>.md`

Write `<body>` only. No frontmatter.

Tell the caller which paths were created.
