---
name: update-codeowners
description: Updates CODEOWNERS entries safely with consistent path and owner formatting. Use when the user asks to add, remove, or modify CODEOWNERS rules, ownership mappings, reviewers, or module maintainers.
---

# Update CODEOWNERS

## Goal
Maintain `CODEOWNERS` accurately while preserving the repository's existing section/comment style.

## Workflow
1. Read the current `CODEOWNERS` file before editing.
2. Identify requested changes as one of:
   - Add new path rule
   - Modify owners for existing path rule
   - Remove obsolete path rule
   - Reorganize section comments (only if requested)
3. Update rules in place instead of creating duplicates for the same path.
4. Keep existing section headers and comment style unless the user asks to refactor structure.
5. Return a concise changelog describing which paths were added, changed, or removed.

## Rule Format
- Use one rule per line: `<path-pattern> <owner1> <owner2> ...`
- Owners must be GitHub handles prefixed with `@`.
- Keep path style consistent with the file (in this repo, path patterns typically start with `/`).
- Do not leave rules with empty owner lists.

## Editing Guidelines
- Prefer minimal edits near related sections.
- If a path already exists, update that line instead of adding a second conflicting line.
- If a new rule logically belongs to an existing section, place it in that section.
- Preserve human-readable grouping and blank lines.
- Keep comments intact unless they are clearly outdated and the user asked for cleanup.

## Validation Checklist
- [ ] Every non-comment, non-empty line has at least one owner.
- [ ] Every owner token starts with `@`.
- [ ] No accidental duplicate rule for the exact same path pattern.
- [ ] Existing comments/sections were preserved unless explicitly changed.

## Example Requests
- "Add `/mllm/models/new_model/ @alice @bob` under models."
- "Change `/core/Storage` owner to `@team-core`."
- "Remove ownership rule for deprecated path `/legacy/`."
