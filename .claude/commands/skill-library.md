---
description: Show and manage the CaP-X evolving skill library of reusable robot control functions.
allowed-tools: Bash, Read
argument-hint: [show|add|remove|promote|extract]
---

# Skill Library: $ARGUMENTS

The CaP-X skill library (`.capx_skills.json`) stores reusable Python functions extracted from successful robot manipulation trials. Skills that appear across multiple tasks are automatically promoted and made available to future code generation.

## Commands

### show (default)
```bash
uv run --no-sync --active python -c "
from capx.skills.library import SkillLibrary
lib = SkillLibrary()
print(lib.summary())
"
```

### add
```bash
uv run --no-sync --active python -c "
from capx.skills.library import SkillLibrary
lib = SkillLibrary()
lib.add_skill('SKILL_NAME', '''SKILL_CODE''')
lib.save()
print('Added. Total skills:', len(lib.skills))
"
```

### extract
Extract all functions from a Python file into the skill library:
```bash
uv run --no-sync --active python -c "
from capx.skills.library import SkillLibrary
from capx.skills.extractor import extract_functions
code = open('PATH_TO_CODE.py').read()
funcs = extract_functions(code)
lib = SkillLibrary()
for name, src in funcs.items():
    lib.add_skill(name, src)
lib.save()
print(f'Extracted {len(funcs)} skills: {list(funcs.keys())}')
"
```

### remove / promote
```bash
uv run --no-sync --active python -c "
from capx.skills.library import SkillLibrary
lib = SkillLibrary()
lib.remove_skill('name')  # or lib.promote('name')
lib.save()
print(lib.summary())
"
```

## How it works
1. During evaluation trials with `evolve_skill_library: true`, functions from successful trials are auto-extracted
2. Skills appearing in 2+ successful trials across different tasks are auto-promoted
3. Promoted skills are injected into the code execution namespace for future trials
