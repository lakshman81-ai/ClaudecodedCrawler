# EduCrawler: Complete Technical Specification
## Agentic Educational Content Crawler for Grade 8 Study Materials

**Version:** 1.0.0  
**Date:** January 2026  
**Author:** StudyHub Development Team  
**Target Platform:** Google Antigravity + NotebookLM

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Models](#3-data-models)
4. [Source Handler Specifications](#4-source-handler-specifications)
5. [State Machine Definition](#5-state-machine-definition)
6. [NotebookLM Integration](#6-notebooklm-integration)
7. [Output Templates](#7-output-templates)
8. [Configuration Schema](#8-configuration-schema)
9. [Error Handling](#9-error-handling)
10. [Testing Specifications](#10-testing-specifications)

---

## 1. Executive Summary

### 1.1 Purpose

EduCrawler is an agentic web crawler designed to extract, aggregate, and compile educational content for Grade 8 students. It scrapes trusted educational platforms (Khan Academy, Byjus, Vedantu), extracts YouTube transcripts, and uses NotebookLM for AI-powered content synthesis.

### 1.2 Key Features

- **Multi-source Aggregation**: Parallel extraction from 4+ educational platforms
- **Intelligent Planning**: LLM-driven source prioritization based on query type
- **NotebookLM Integration**: Automated compilation via DOM manipulation
- **Three Output Formats**: Study Material, Questionnaire, Handout
- **Graceful Degradation**: Cascading fallback with quality gates

### 1.3 Technology Stack

| Component | Technology |
|-----------|------------|
| Execution Environment | Google Antigravity (Gemini 3) |
| Browser Automation | Antigravity Browser Subagent + Playwright |
| Language | Python 3.11+ with async/await |
| Data Validation | Pydantic v2 |
| HTML Processing | BeautifulSoup4 + markdownify |
| YouTube Transcripts | youtube-transcript-api |
| Content Synthesis | Google NotebookLM |
| Output Rendering | Jinja2 Templates |

### 1.4 Supported Subjects

| Subject | Topics Covered |
|---------|----------------|
| Physics | Motion, Force, Work, Energy, Light, Sound |
| Chemistry | Atomic Structure, Chemical Reactions, Acids & Bases |
| Mathematics | Algebra, Geometry, Probability, Exponents |
| Biology | Cell Structure, Human Body, Ecosystems |
| History | Ancient Civilizations, Mughal Empire, Modern History |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EDUCRAWLER SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    PRESENTATION LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ Antigravity │  │ NotebookLM  │  │ Rich HTML Output            │  │   │
│  │  │ Window      │  │ Document    │  │ (Study/Quiz/Handout)        │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                        │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    APPLICATION LAYER                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Crawler     │  │ Content     │  │ NotebookLM  │  │ Output     │  │   │
│  │  │ Agent       │  │ Processor   │  │ Integrator  │  │ Renderer   │  │   │
│  │  │             │  │             │  │             │  │            │  │   │
│  │  │ • Planner   │  │ • HTML→MD   │  │ • Auth Mgr  │  │ • Study    │  │   │
│  │  │ • Dispatcher│  │ • Dedup     │  │ • Source Add│  │ • Quiz     │  │   │
│  │  │ • State Mgr │  │ • Aggregator│  │ • Compiler  │  │ • Handout  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                     ▲                                        │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA ACCESS LAYER                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Khan Academy│  │ Byjus       │  │ Vedantu     │  │ YouTube    │  │   │
│  │  │ Handler     │  │ Handler     │  │ Handler     │  │ Handler    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌─────────────────────────────────────────────┐   │   │
│  │  │ Google      │  │ Rate Limiter + Retry Manager                 │   │   │
│  │  │ Search      │  │ (Shared Infrastructure)                      │   │   │
│  │  │ Handler     │  └─────────────────────────────────────────────┘   │   │
│  │  └─────────────┘                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| **Crawler Agent** | Orchestrates entire workflow | CrawlRequest | AggregatedContent |
| **Planner** | Analyzes query, prioritizes sources | Query string | ExecutionPlan |
| **Dispatcher** | Manages parallel execution | ExecutionPlan | List[ExtractedContent] |
| **Source Handlers** | Site-specific extraction | URL + Config | ExtractedContent |
| **Content Processor** | Cleans, normalizes, deduplicates | Raw content | ProcessedContent |
| **NotebookLM Integrator** | Automates compilation | Sources list | CompiledContent |
| **Output Renderer** | Formats final output | CompiledContent | HTML/Rich Text |

### 2.3 Data Flow Sequence

```
User Input                    Processing                         Output
─────────────────────────────────────────────────────────────────────────────
                                                                  
"Grade 8, Physics,           ┌─────────────┐                     
 Work Done"                  │   PLANNER   │                     
      │                      │             │                     
      ▼                      │ Analyze     │                     
┌─────────────┐              │ query type: │                     
│ CrawlRequest│─────────────▶│ CONCEPTUAL  │                     
│             │              │             │                     
│ grade: 8    │              │ Priority:   │                     
│ subject:    │              │ 1. Khan     │                     
│  Physics    │              │ 2. Byjus    │                     
│ topic:      │              │ 3. Vedantu  │                     
│  Work Done  │              │ 4. YouTube  │                     
│ output:     │              └──────┬──────┘                     
│  StudyMat   │                     │                            
└─────────────┘                     ▼                            
                             ┌─────────────┐                     
                             │ DISPATCHER  │                     
                             │             │                     
                             │ Parallel    │                     
                             │ execution:  │                     
                             └──────┬──────┘                     
                 ┌──────────────────┼──────────────────┐         
                 ▼                  ▼                  ▼         
          ┌──────────┐       ┌──────────┐       ┌──────────┐    
          │ Khan     │       │ Byjus    │       │ YouTube  │    
          │ Handler  │       │ Handler  │       │ Handler  │    
          └────┬─────┘       └────┬─────┘       └────┬─────┘    
               │                  │                  │           
               ▼                  ▼                  ▼           
          ExtractedContent  ExtractedContent  ExtractedContent   
               │                  │                  │           
               └──────────────────┼──────────────────┘           
                                  ▼                              
                           ┌─────────────┐                       
                           │ PROCESSOR   │                       
                           │             │                       
                           │ • Merge     │                       
                           │ • Dedup     │                       
                           │ • Validate  │                       
                           └──────┬──────┘                       
                                  ▼                              
                           ┌─────────────┐                       
                           │ NOTEBOOKLM  │                       
                           │ INTEGRATOR  │                       
                           │             │      ┌─────────────┐  
                           │ • Add srcs  │─────▶│ Study       │  
                           │ • Compile   │      │ Material    │  
                           │ • Extract   │      │ (HTML)      │  
                           └─────────────┘      └─────────────┘  
```

### 2.4 Execution Environment Configuration

```yaml
# antigravity_config.yaml
platform: google_antigravity
version: "1.0"

security:
  js_execution_policy: "allow_list"
  allowed_domains:
    - khanacademy.org
    - byjus.com
    - vedantu.com
    - youtube.com
    - notebooklm.google.com
  blocked_domains:
    - facebook.com
    - twitter.com

browser_subagent:
  headless: false  # Required for NotebookLM
  viewport:
    width: 1280
    height: 800
  user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  
timeouts:
  page_load: 30000  # 30 seconds
  element_wait: 10000
  network_idle: 5000

logging:
  level: DEBUG
  format: json
  destination: antigravity_console
```

### 2.5 Error Handling Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FOUR-TIER ERROR HANDLING STRATEGY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 1: PRIMARY SOURCE                                                      │
│  ├── Attempt extraction from primary source (based on priority)             │
│  ├── Wait for JS rendering + MathJax (if required)                          │
│  └── If success → continue to processing                                    │
│                                                                              │
│  TIER 2: CACHED VERSION (on failure)                                        │
│  ├── Check for cached/archived version                                      │
│  ├── Try web.archive.org or Google Cache                                    │
│  └── If success → continue with cached content                              │
│                                                                              │
│  TIER 3: ALTERNATIVE SOURCE (on failure)                                    │
│  ├── Switch to next priority source                                         │
│  ├── Apply same extraction logic                                            │
│  └── If success → continue to processing                                    │
│                                                                              │
│  TIER 4: DEGRADED EXTRACTION (on failure)                                   │
│  ├── Extract text-only content (no JS rendering)                           │
│  ├── Use basic HTML parsing                                                 │
│  └── Mark content as "degraded" quality                                     │
│                                                                              │
│  CIRCUIT BREAKER                                                             │
│  ├── Track failures per domain (threshold: 3 consecutive)                  │
│  ├── Open circuit → skip domain for 5 minutes                              │
│  └── Half-open → try single request after cooldown                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Models

All data models use Pydantic v2 for validation and serialization. See `models/` directory for complete implementations.

### 3.1 Core Models Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA MODEL HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  REQUEST MODELS                                                              │
│  ├── CrawlRequest          # User input with validation                     │
│  ├── CrawlConfig           # Execution configuration                        │
│  └── SourcePriority        # Source ordering preferences                    │
│                                                                              │
│  CONTENT MODELS                                                              │
│  ├── ExtractedContent      # Raw content from single source                 │
│  │   ├── Concept           # Individual concept/topic                       │
│  │   ├── Formula           # Mathematical formula with LaTeX                │
│  │   ├── Example           # Worked example with steps                      │
│  │   └── Exercise          # Practice problem                               │
│  ├── ProcessedContent      # Cleaned and normalized content                 │
│  └── AggregatedContent     # Merged from multiple sources                   │
│                                                                              │
│  OUTPUT MODELS                                                               │
│  ├── OutputDocument        # Base output class                              │
│  ├── StudyMaterial         # 5E/LES structured study guide                  │
│  ├── Questionnaire         # 50-question tiered assessment                  │
│  │   ├── MCQuestion        # Multiple choice                                │
│  │   ├── FillBlankQuestion # Fill in the blank                              │
│  │   ├── TrueFalseQuestion # True/False                                     │
│  │   └── DetailedQuestion  # Long answer / case study                       │
│  └── Handout               # 3-zone quick reference                         │
│                                                                              │
│  STATE MODELS                                                                │
│  ├── CrawlState            # Current execution state                        │
│  ├── ExecutionPlan         # Planned actions                                │
│  └── CrawlResult           # Final result with metadata                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Relationships

```
CrawlRequest ──────────────────────────────────────────────────────┐
     │                                                              │
     ▼                                                              │
ExecutionPlan ────┬─────────────────────────────────────────┐      │
     │            │                                          │      │
     │            ▼                                          │      │
     │     SourceTask[0..n]                                  │      │
     │            │                                          │      │
     │            ▼                                          │      │
     │     ExtractedContent[0..n] ───┐                      │      │
     │            │                   │                      │      │
     │            ▼                   │                      │      │
     │     ProcessedContent[0..n]    │                      │      │
     │            │                   │                      │      │
     │            ▼                   ▼                      │      │
     │     AggregatedContent ◄───────┘                      │      │
     │            │                                          │      │
     │            ▼                                          │      │
     │     NotebookLMSession                                │      │
     │            │                                          │      │
     │            ▼                                          │      │
     │     CompiledContent                                   │      │
     │            │                                          │      │
     │            ▼                                          ▼      │
     └────▶ OutputDocument (Study | Quiz | Handout) ◄──────────────┘
                  │
                  ▼
            CrawlResult
```

