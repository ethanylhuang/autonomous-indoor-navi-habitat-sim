# Orchestrator Protocol

This document defines how the main agent (you) spawns and manages the Three Hats pipeline. Follow it exactly.

## Core Principle

Each sub-agent is **blind** to everything except what you explicitly pass it. You are the only bridge between agents. This means:
- You control what each agent sees
- You are responsible for artifact completeness
- You must never leak conversation history, prior reasoning, or context from other agents

## Spawning Rules

### What Each Agent Receives

| Agent | Reads its own template | Task description | Prior artifacts | Source files | Conversation history |
|-------|----------------------|------------------|-----------------|--------------|---------------------|
| Architect | Yes (`.claude/agents/architect.md`) | Yes | None | Relevant existing files | **NO** |
| Builder | Yes (`.claude/agents/builder.md`) | No (gets it via architect artifact) | Architect artifact | Files needed for implementation | **NO** |
| Shield | Yes (`.claude/agents/shield.md`) | No (gets it via artifacts) | Architect + Builder artifacts | Changed/created files only | **NO** |

### How to Construct Each Prompt

**Architect prompt:**
```
Read your instructions from: .claude/agents/architect.md
Read the project plan from: Implementation Plan/PROJECT_PLAN.md

TASK: [one paragraph describing what to design]

EXISTING FILES TO CONSIDER:
[list specific file paths the architect should read — only files relevant to the design]
```

**Builder prompt:**
```
Read your instructions from: .claude/agents/builder.md

ARCHITECT ARTIFACT (approved by Traso):
[paste the full architect output — all sections]

SOURCE FILES:
[list specific file paths the builder needs to read or modify]
```

**Shield prompt:**
```
Read your instructions from: .claude/agents/shield.md

ARCHITECT ARTIFACT:
[paste the full architect output — all sections]

BUILDER ARTIFACT:
[paste the full builder output — all sections]

CHANGED FILES:
[list the exact files that were created or modified by the builder]
```

### What You Must NOT Do

- Do NOT summarize or paraphrase artifacts — pass them verbatim
- Do NOT add your own commentary or "helpful context" to agent prompts
- Do NOT tell an agent what another agent "was thinking"
- Do NOT pass file contents inline if the agent can read the file itself — just pass the path
- Do NOT combine two pipeline stages into one agent spawn
- Do NOT pass the CLAUDE.md to agents — their templates already contain the project context they need

## Pipeline Execution

### Step 1: Architect

1. Spawn `Task(subagent_type=Plan)` with architect prompt
2. Receive architect artifact
3. **Validate artifact completeness** — must contain all required sections:
   - `ASSUMPTIONS`, `IN_SCOPE`, `OUT_OF_SCOPE`, `DESIGN`, `RISKS`, `ACCEPTANCE_CRITERIA`
4. If any section is missing or empty: re-spawn architect with a note about what's missing
5. **Present artifact to Traso** — display the full output
6. **Wait for approval** — do not proceed until Traso explicitly approves
7. If Traso requests changes: re-spawn architect with the feedback

### Step 2: Builder

1. Only after Traso approves the architect artifact
2. Spawn `Task(subagent_type=general-purpose)` with builder prompt
3. Receive builder artifact
4. **Validate artifact completeness** — must contain all required sections:
   - `PATCH_PLAN`, `IMPLEMENTATION`, `CHANGED_FILES`, `VERIFY_STEPS`, `ROLLBACK_PLAN`
5. If any section is missing: re-spawn builder
6. **Validate design compliance** — skim the implementation to check it matches the architect's DESIGN section
7. If it deviates: re-spawn builder with a note about the deviation

### Step 3: Shield

1. Only after builder artifact is complete and design-compliant
2. Spawn `Task(subagent_type=general-purpose)` with shield prompt
3. Receive shield artifact
4. **Validate artifact completeness** — must contain all required sections:
   - `PASS_CRITERIA`, `FAILURE_MODES`, `REMAINING_RISK`, `ACTION_ITEMS`, `REPRO_STEPS`
5. **Check ACTION_ITEMS for blockers:**
   - If **no blockers**: report success to Traso
   - If **blockers found**: spawn a new Builder with the blocker details, then re-run Shield
   - If **design flaw identified**: go back to Architect with the feedback
6. Present shield results to Traso

## Failure Handling

### Architect produces incomplete artifact
→ Re-spawn architect. Tell it which sections are missing. Do not fill them in yourself.

### Builder deviates from design
→ Re-spawn builder. Paste the specific DESIGN section it violated. Do not patch the code yourself.

### Shield finds blockers
→ Spawn a **new** Builder with:
- The original architect artifact
- The shield's ACTION_ITEMS (blockers only)
- The current source files (post first builder pass)
→ Then re-run Shield on the updated code.

### Shield finds design flaw
→ Go back to Architect with:
- The original task description
- The shield's finding explaining why the design is flawed
- Existing source files
→ Restart the full pipeline from the new architect artifact.

### Agent hits context limits or fails
→ Break the task into smaller sub-tasks. Each sub-task gets its own full pipeline pass.

## Artifact Storage

After each successful pipeline completion:
- Architect artifact: do not persist (lives in conversation)
- Builder artifact: the code is already written to files
- Shield artifact: do not persist unless ACTION_ITEMS remain open

## When to Skip the Pipeline

Use your judgment, but default to running it. Skip only for:
- Single-line fixes, typos, config value changes
- Adding a comment or docstring
- Renaming a variable
- Any change touching 1 file with < 20 lines changed

If in doubt, ask Traso.
