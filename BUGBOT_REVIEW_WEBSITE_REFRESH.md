# BugBot Review: Website Refresh PR

**PR**: feat: refresh BrainLayer website  
**Branch**: `feat/update-brainlayer-site`  
**Date**: 2026-03-30  
**Reviewer**: @bugbot

---

## Executive Summary

✅ **Build & Lint**: PASS  
✅ **TypeScript**: PASS  
✅ **Accessibility**: PASS  
⚠️ **Content Accuracy**: CRITICAL ISSUE FOUND

---

## Critical Issues

### 🔴 CRITICAL: Tool Count Mismatch

**Location**: Multiple files across the site  
**Severity**: Critical - Misleading marketing claim

**Issue**: The website claims "6 working tools" but the actual MCP server exposes **11 tools** (with 3 deprecated).

**Evidence**:

From `/workspace/CLAUDE.md` line 70:
```
11 tools: brain_search, brain_store, brain_recall, brain_entity, brain_expand, 
brain_update, brain_digest, brain_get_person, brain_tags, brain_supersede, brain_archive
```

From `/workspace/src/brainlayer/mcp/__init__.py` lines 269-1062:
- 11 tools are registered in `list_tools()`
- 3 are deprecated but still exposed (brain_expand, brain_update, brain_tags)
- 8 are fully functional

**Website claims "6 tools" in**:
- `site/components/hero.tsx` line 59: "6 working MCP tools"
- `site/components/tools.tsx` line 54: "Six working tools. One memory layer."
- `site/app/layout.tsx` line 27: "Six working tools"
- `site/app/layout.tsx` line 40: "six-tool surface"
- `site/components/cta.tsx` line 33: "Six working tools"
- `site/app/docs/page.tsx` line 207: "six-tool memory surface"

**Actual tool count**:
- **11 total tools** registered in MCP server
- **8 fully functional** (brain_expand, brain_update, brain_tags are deprecated stubs)
- **3 primary tools** per server instructions (brain_recall, brain_store, brain_digest)

**Recommendation**: Update all "6 tools" references to either:
- "11 MCP tools" (accurate count of all exposed tools)
- "8 working tools" (excludes deprecated stubs)
- "3 primary tools + 8 advanced" (matches server's own description)

---

## Medium Issues

### ⚠️ MEDIUM: Inconsistent Tool Descriptions

**Location**: `site/components/tools.tsx`

The site lists 6 tools (3 core + 3 advanced) but omits:
- `brain_get_person` (fully functional)
- `brain_supersede` (fully functional)
- `brain_archive` (fully functional)
- `brain_enrich` (fully functional, added recently)
- `brain_update` (deprecated but still exposed)
- `brain_tags` (deprecated but still exposed)
- `brain_expand` (deprecated but still exposed)

**Current site listing**:
```
Core:
- brain_search
- brain_store
- brain_recall

Advanced:
- brain_entity
- brain_expand (deprecated!)
- brain_digest
```

**Recommendation**: Either list all 11 tools or clarify that the site shows "primary tools" and link to full docs.

---

## Low Issues

### ℹ️ LOW: Terminal Animation Timing

**Location**: `site/components/terminal.tsx` line 277

The typing animation uses hardcoded delays that may not sync well with the prompt typing speed on slower devices.

**Recommendation**: Consider using `prefers-reduced-motion` media query (already handled on line 258) and potentially adjusting timing constants.

---

### ℹ️ LOW: Missing Alt Text Context

**Location**: `site/components/integrations.tsx` line 57

Logo images use generic alt text like "Claude Code" but could be more descriptive for screen readers.

**Current**: `alt="Claude Code"`  
**Better**: `alt="Claude Code logo - MCP client integration"`

---

## Positive Findings

✅ **Accessibility**: Proper ARIA labels, focus states, keyboard navigation  
✅ **Performance**: Static generation, optimized images, no blocking resources  
✅ **Code Quality**: Clean TypeScript, proper React patterns, no linter errors  
✅ **Responsive Design**: Mobile-first, proper breakpoints, fluid typography  
✅ **Animation**: Respects `prefers-reduced-motion`  
✅ **SEO**: Proper metadata, OpenGraph tags, semantic HTML

---

## Test Results

```bash
$ npm run lint
✅ PASS (0 errors, 0 warnings)

$ npm run build
✅ PASS
- TypeScript compilation: 1999ms
- Static pages generated: 5/5
- Build time: 5.6s
```

---

## Recommendations

### Immediate (Block Merge)

1. **Fix tool count claims** - Update all "6 tools" to accurate count
2. **Update tools.tsx** - Either list all tools or add disclaimer

### Before Production Deploy

3. Review and update tool descriptions to match current MCP server instructions
4. Consider adding a "deprecated tools" section or note in docs

### Nice to Have

5. Improve logo alt text for accessibility
6. Add link to full tool documentation from tools section

---

## Conclusion

The website refresh is **high quality** from a technical perspective (build, lint, accessibility, performance all pass), but contains a **critical content accuracy issue** regarding the tool count.

**Recommendation**: Fix the tool count discrepancy before merging to production.

---

**Review completed**: 2026-03-30  
**Reviewer**: @bugbot (autonomous code review agent)
