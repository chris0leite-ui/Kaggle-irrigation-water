# Install the `kaggle-comp` skill

Paste the prompt below into a fresh Claude Code session (anywhere —
the skill is user-level, so the working directory doesn't matter
for installation). The skill source lives on GitHub; the container
fetches it and copies it into `~/.claude/skills/`.

## The install prompt

````
Install the kaggle-comp skill from my Kaggle-irrigation-water GitHub repo to ~/.claude/skills/.

Container is fresh; pull from GitHub.

1. **Fetch**: `git clone --depth 1 https://github.com/chris0leite-ui/Kaggle-irrigation-water.git /tmp/_skill_src`
   (or use the GitHub MCP server if HTTPS clone isn't authenticated;
   repo is `chris0leite-ui/Kaggle-irrigation-water`, branch `main`).

2. **Install**: `mkdir -p ~/.claude/skills && cp -r /tmp/_skill_src/writeup/skill/kaggle-comp ~/.claude/skills/`

3. **Verify** `ls ~/.claude/skills/kaggle-comp/` shows: SKILL.md, kickoff-runbook.md,
   kickoff-bash.md, kickoff.md, guardrails.md, personas.md, loops.md, self-improvement.md,
   do-and-dont.md, examples/, templates/.

4. **Pre-flight kaggle**: `kaggle --version` works (`pip install -q kaggle` if not),
   `KAGGLE_API_TOKEN` is set (ask me once, redacted, if not).

5. **Cleanup** /tmp/_skill_src.

6. **Report** in 3 lines: install path, kaggle CLI version, token status.
   Then **wait** for me to say "let's do the kickoff".

Do NOT auto-trigger the kickoff. Install + verify only.
````

## After install

1. `cd` into an empty directory (or wherever you want the new-comp
   repo to live; the kickoff scaffolds in place).
2. Open a fresh Claude Code session there.
3. Say "**let's do the kickoff**" (or "start a new comp" / "kickoff").
4. The skill's [kickoff-runbook.md](kickoff-runbook.md) takes over —
   Q1 asks for the slug; everything else flows automatically.

## Re-installing after skill updates

The skill evolves comp-over-comp via the friction-distillation loop
in [self-improvement.md](self-improvement.md). When the skill files
on `main` change, re-run the install prompt above. It overwrites
`~/.claude/skills/kaggle-comp/` cleanly.

## Why user-level instead of project-level

`~/.claude/skills/kaggle-comp/` covers every future comp from a single
install. A project-level `.claude/skills/` would scope the skill to
one comp, defeating the cross-comp reuse goal.
