# EduCrawler: Phase 1 Specifications Complete

## Overview

This directory contains the complete Phase 1 technical specifications for the EduCrawler project - an agentic web crawler for Grade 8 educational content.

**Status:** ✅ Phase 1 Complete - Ready for Phase 2 Implementation

---

## Directory Structure

```
educrawler_specs/
│
├── 📄 SPECIFICATION.md           # Main architecture & data flow
├── 📄 SPECIFICATION_PART2.md     # Error handling & testing specs
├── 📄 README.md                  # This file
│
├── 📁 models/                    # Pydantic data models
│   ├── __init__.py              # Model exports
│   ├── request.py               # CrawlRequest, CrawlConfig, SourcePriority
│   ├── content.py               # ExtractedContent, Concept, Formula, Example
│   ├── output.py                # StudyMaterial, Questionnaire, Handout
│   └── state.py                 # CrawlState, ExecutionPlan, StateMachine
│
├── 📁 handlers/                  # Source handler specifications
│   ├── base.py                  # AbstractSourceHandler base class
│   ├── source_specs.py          # CSS selectors, rate limits per source
│   ├── youtube.py               # YouTube transcript handler (API-based)
│   └── notebooklm_integration.py # NotebookLM browser automation
│
├── 📁 templates/                 # Jinja2 HTML templates
│   ├── study_material.html      # 5-page study guide template
│   ├── questionnaire.html       # 50-question assessment template
│   └── handout.html             # Single-page reference template
│
├── 📁 config/                    # Configuration files
│   └── config.yaml              # Complete configuration schema
│
└── 📁 tests/                     # Test specifications (empty, for Phase 2)
    └── (to be implemented)
```

---

## What's Included

### 1. Data Models (`models/`)

Complete Pydantic v2 models with validation:

| Model | Description | Key Features |
|-------|-------------|--------------|
| `CrawlRequest` | User input validation | Grade, subject, topic, output type |
| `CrawlConfig` | Execution settings | Rate limits, timeouts, quality gates |
| `ExtractedContent` | Raw content from source | Concepts, formulas, examples, exercises |
| `AggregatedContent` | Merged multi-source content | Deduplication, quality scoring |
| `StudyMaterial` | Study guide output | 5E/LES framework, Cornell notes |
| `Questionnaire` | 50-question assessment | Tier 1/2/3 with A-R questions |
| `Handout` | Quick reference | Formula table, visual map, pro tips |
| `CrawlState` | State machine | IDLE → PLANNING → CRAWLING → ... |
| `ExecutionPlan` | Planned execution | Source tasks with priorities |

### 2. Handler Specifications (`handlers/`)

| Handler | Type | Key Details |
|---------|------|-------------|
| Khan Academy | Browser + JS | MathJax wait, `.article-content` selector |
| Byjus | Browser + JS | Cloudflare protection, 3s rate limit |
| Vedantu | Browser + JS | PDF downloads available |
| YouTube | API | `youtube-transcript-api`, proxy rotation |
| Google Search | Browser | Fallback only, 5s rate limit |
| NotebookLM | Browser automation | Playwright, persistent auth |

### 3. Output Templates (`templates/`)

All templates follow the validation checklist:

- **Study Material**: A4 landscape, Cornell notes, 3-4 concept boxes/page
- **Questionnaire**: Exactly 50 questions (20-20-10), 6+ A-R questions
- **Handout**: 3 zones, ≤5 colors, formula table + visual map + pro tips

### 4. Configuration (`config/`)

Complete YAML configuration with:
- Source-specific settings
- Rate limiting defaults
- Browser configuration
- NotebookLM settings
- Quality gates
- Subject-specific color palettes

---

## Phase 2: Implementation Guide

### Recommended Execution Environment

| Phase | Environment | Why |
|-------|-------------|-----|
| **Core modules** | Claude Code | Local testing, file management |
| **Browser automation** | Gemini in Antigravity | Native browser subagent |
| **NotebookLM integration** | Gemini in Antigravity | Same Google ecosystem |
| **Debugging** | Both as needed | Depends on issue type |

### Implementation Order

```
1. ✅ [DONE] Data models (models/)
2. ⬜ Utility modules (rate limiter, retry logic)
3. ⬜ YouTube handler (easiest, API-based)
4. ⬜ Content processor (HTML→MD, deduplication)
5. ⬜ Output renderers (template rendering)
6. ⬜ Khan Academy handler (best documented)
7. ⬜ Byjus/Vedantu handlers (similar to Khan)
8. ⬜ NotebookLM integrator (browser automation)
9. ⬜ Orchestration agent (Plan-and-Execute)
10. ⬜ Integration testing
```

### Key Instructions for Claude Code

When implementing in Claude Code, provide this context:

```markdown
## Context
I have complete specifications in /path/to/educrawler_specs/
Please read the relevant spec files before implementing:
- models/__init__.py for all data models
- handlers/source_specs.py for CSS selectors
- config/config.yaml for configuration

## Implementation Requirements
- Use async/await for all I/O
- Add type hints to ALL functions
- Include docstrings
- Run ruff check after each file
- Test each module before moving on
```

### Key Instructions for Antigravity

When implementing browser automation in Antigravity:

```markdown
## Context
I need to implement browser-based source handlers.
The specifications are in handlers/source_specs.py.

## Tasks
1. Implement KhanAcademyHandler.fetch() and extract()
2. Wait for MathJax rendering (2.5 seconds)
3. Use CSS selectors from source_specs.py
4. Handle rate limiting (2s minimum between requests)
```

---

## Validation Checklist Summary

### Study Material
- [ ] 5 pages following pedagogical framework
- [ ] 3-4 concept boxes per page (💡⚠️🌍🔍)
- [ ] 1-2 images per page
- [ ] Unicode formulas (no images)
- [ ] Cornell notes format

### Questionnaire
- [ ] **EXACTLY 50 questions** (zero tolerance)
- [ ] Tier 1: 20 questions (10 MCQ, 5 FIB, 5 T/F)
- [ ] Tier 2: 20 questions (application-based)
- [ ] Tier 3: 10 questions (6+ assertion-reasoning)
- [ ] Complete answer key with explanations

### Handout
- [ ] Zone 1: Formula table
- [ ] Zone 2: Visual map (mind map/timeline/flowchart)
- [ ] Zone 3: Pro tips (memory tricks, don't confuse)
- [ ] ≤5 colors
- [ ] Single A4 landscape page

---

## Dependencies

```bash
# Python packages
pip install pydantic aiohttp playwright beautifulsoup4
pip install markdownify youtube-transcript-api jinja2
pip install minhashlsh  # For deduplication

# Playwright browsers
playwright install chromium

# Optional: wkhtmltopdf for PDF generation
```

---

## Quick Start for Phase 2

1. **Copy models to your project:**
   ```bash
   cp -r models/ /your/project/src/models/
   ```

2. **Install dependencies:**
   ```bash
   pip install pydantic aiohttp beautifulsoup4 markdownify
   ```

3. **Test model imports:**
   ```python
   from models import CrawlRequest, Subject, OutputType
   
   request = CrawlRequest(
       grade=8,
       subject=Subject.PHYSICS,
       topic="Work Done",
       output_type=OutputType.STUDY_MATERIAL
   )
   print(request.model_dump_json(indent=2))
   ```

4. **Implement handlers using base class:**
   ```python
   from handlers.base import AbstractSourceHandler
   
   class MyHandler(AbstractSourceHandler):
       SOURCE_NAME = "my_source"
       # ... implement abstract methods
   ```

---

## Support

For questions about these specifications:
- Review the architecture in `SPECIFICATION.md`
- Check error handling in `SPECIFICATION_PART2.md`
- See CSS selectors in `handlers/source_specs.py`
- Review configuration options in `config/config.yaml`

---

**Phase 1 Completed:** January 2026  
**Next Phase:** Implementation in Claude Code + Antigravity
