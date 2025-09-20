<# ARCHIVIO: lo script originale per la creazione di progetti è stato spostato in
   `scripts/create_project.ps1`.

   Questo file rimane come riferimento. Per eseguire lo script, usa:
     pwsh .\scripts\create_project.ps1

#>
*.rar
*.7z
*.bak
*.old
"@
$gitignore | Set-Content $GitignorePath
}

# Create starter PRD template
$PRDFile = "$ProjectPath\docs\requirements\PRD_${ProjectName}.md"
if (!(Test-Path $PRDFile)) {
$prdContent = @"
# Product Requirements Document (PRD) — $ProjectName

## 1. Overview
Brief summary of the project, its purpose, and the problems it solves.

## 2. Goals & Objectives
- High-level goals
- Measurable objectives

## 3. Stakeholders
- Product Owner:
- Tech Lead:
- Team Members:
- Other Stakeholders:

## 4. Scope
### In Scope
- Features & functionality to deliver.

### Out of Scope
- Features excluded from this release.

## 5. User Stories
| ID | As a... | I want... | So that... |
|----|---------|-----------|------------|
| 1  |         |           |            |

## 6. Functional Requirements
- Feature-by-feature breakdown
- APIs, integrations, or systems involved

## 7. Technical Requirements
- Platforms, frameworks, libraries
- Performance requirements

## 8. Design
- Link to wireframes, mockups, or diagrams in docs/architecture/

## 9. Data
- Input sources, formats
- Expected outputs

## 10. Risks & Assumptions
- Risks and mitigations
- Key assumptions

## 11. Milestones & Roadmap
| Milestone | Description | ETA |
|-----------|-------------|-----|
| Alpha     |             |     |
| Beta      |             |     |
| GA        |             |     |

---

_Last updated: $(Get-Date -Format yyyy-MM-dd)_
"@
$prdContent | Set-Content $PRDFile
}

# Create project context file for Claude
$ProjectContextFile = "$ProjectPath\docs\PROJECT_CONTEXT.md"
if (!(Test-Path $ProjectContextFile)) {
$contextContent = @"
# Project Context for Claude AI

## Project Overview
**Name:** $ProjectName  
**Created:** $(Get-Date -Format yyyy-MM-dd)  
**Type:** [Web App / Mobile App / Game / API / Data Science / AI/ML / Other]  
**Tech Stack:** [To be defined]  

## Quick Navigation
- 📋 Requirements: \`docs/requirements/PRD_${ProjectName}.md\`
- 🏗️ Architecture: \`docs/architecture/\`
- 💻 Main Code: \`src/\`
- 🧪 Experiments: \`experiments/\` and \`sandbox/\`
- 📊 Data: \`data/\`
- 🤖 AI/ML: \`models/\` and \`src/agents/\`
- ✅ Tests: \`tests/\`
- 🚀 Deployment: \`deployment/\`

## Current Status
- [ ] Project initialized
- [ ] Requirements defined
- [ ] Architecture designed
- [ ] Development started
- [ ] Testing implemented
- [ ] Deployment configured

## Key Files & Locations
| Purpose | Location | Notes |
|---------|----------|-------|
| Main application code | \`src/\` | Core business logic |
| API/Backend | \`src/backend/\` | Server-side code |
| Frontend/UI | \`src/frontend/\` | User interface |
| Shared code | \`src/shared/\` | Reusable components |
| Database configs | \`configs/database/\` | Connection strings, schemas |
| Environment configs | \`configs/environments/\` | Dev, staging, prod settings |
| Claude collaboration | \`sandbox/claude_sessions/\` | Save our work sessions |
| Quick prototypes | \`prototypes/\` | Proof of concepts |
| Research & spikes | \`research_spikes/\` | Technical investigations |

## Development Guidelines
1. **For New Features:** Start in \`prototypes/\` or \`sandbox/\`, then move to \`src/\`
2. **For Research:** Use \`research_spikes/\` and document findings
3. **For Data Work:** Raw data in \`data/raw/\`, processed in \`data/processed/\`
4. **For AI/ML:** Models in \`models/\`, training scripts in \`src/agents/\`
5. **For Documentation:** Keep \`docs/\` updated with architecture decisions

## Claude Collaboration Notes
- Save important code snippets and solutions in \`sandbox/claude_sessions/\`
- Document architectural decisions in \`docs/architecture/\`
- Use \`experiments/\` for testing new approaches
- Keep a running log of what works/doesn't work

---
_This file helps Claude understand your project structure and current context._
"@
$contextContent | Set-Content $ProjectContextFile
}
$ChangelogFile = "$ProjectPath\docs\changelogs\CHANGELOG.md"
if (!(Test-Path $ChangelogFile)) {
$changelogContent = @"
# Changelog — $ProjectName

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- Initial project structure

### Changed
- Nothing yet

### Deprecated
- Nothing yet

### Removed
- Nothing yet

### Fixed
- Nothing yet

### Security
- Nothing yet

---

## [0.1.0] - $(Get-Date -Format yyyy-MM-dd)
### Added
- Project scaffold created

---
"@
$changelogContent | Set-Content $ChangelogFile
}

Write-Host ""
Write-Host "Project '$ProjectName' scaffolded at: $ProjectPath"
Write-Host "PRD template: $PRDFile"
Write-Host "CHANGELOG template: $ChangelogFile"
Write-Host "Project context for Claude: $ProjectContextFile"
Write-Host "Base development folder: $BasePath"