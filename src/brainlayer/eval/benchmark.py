"""Ranx-based benchmark helpers for BrainLayer search evaluation."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("IR_DATASETS_HOME", "/tmp/ir_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ranx import Qrels, Run, compare, evaluate

from brainlayer._helpers import _escape_fts5_query

DEFAULT_RUN_METRICS = ["ndcg@10", "recall@20", "map@10", "mrr"]
DEFAULT_COMPARE_METRICS = ["ndcg@10", "recall@20"]
MINED_FRUSTRATION_QUERY_SUITE: list[tuple[str, str]] = [
    (
        "frustration_001",
        '{\'file_path\': \'/Users/etanheyman/Desktop/Gits/domica/apps/public/src/components/layout/Header/index.tsx\', \'old_string\': \'function MegaMenuDropdown({ item }: { item: NavItem }) {\\n  const [isOpen, setIsOpen] = useState(false);\\n\\n  if (!item.megaMenu) {\\n    return (\\n      <Link\\n        href={item.href}\\n        className="px-2 py-1 rounded text-sm font-medium text-[#111827] hover:bg-grey-100 transition-colors"\\n      >\\n        {item.label}\\n      </Link>\\n    );\\n  }\\n\\n  return (\\n    <div\\n      className="relative"\\n      onMouseEnter={() => setIsOpen(true)}\\n      onMouseLeave={() => setIsOpen(false)}\\n    >\\n      <button\\n        className="flex items-center gap-1 px-2 py-1 rounded text-sm font-medium text-[#111827] hover:bg-grey-100 transition-colors"\\n      >\\n        <ChevronDown className={cn(\\\'w-3.5 h-3.5 transition-transform\\\', isOpen && \\\'rotate-180\\\')} />\\n        <span>{item.label}</span>\\n      </button>\\n\\n      {/* Invisible bridge to connect button to dropdown */}\\n      {isOpen && <div className="absolute top-full left-0 right-0 h-4" />}\\n\\n      {isOpen && (\\n        <div\\n          className="fixed top-[60px] left-0 right-0 p-6"\\n          onMouseEnter={() => setIsOpen(true)}\\n          onMouseLeave={() => setIsOpen(false)}\\n          style={{\\n            background: \\\'rgba(255, 255, 255, 0.98)\\\',\\n            boxShadow: \\\'0px 8px 32px rgba(0, 0, 0, 0.12)\\\',\\n            backdropFilter: \\\'blur(8px)\\\',\\n          }}\\n          dir="rtl"\\n        >\\n          <div className="max-w-[1200px] mx-auto px-4 md:px-10 lg:px-20">\\n            {/* Main Links Row */}\\n            <div className="flex items-center gap-4 pb-4 mb-4 border-b border-grey-200">\\n              {item.megaMenu.mainLinks.map((link, idx) => (\\n                <Link\\n                  key={idx}\\n                  href={link.href}\\n                  className="text-sm font-semibold text-[#2563EB] hover:underline"\\n                >\\n                  {link.label}\\n                </Link>\\n              ))}\\n            </div>\\n\\n            {/* Columns - split links into chunks of 5 */}\\n            <div className="flex flex-wrap gap-x-10 gap-y-4">\\n              {item.megaMenu.columns.flatMap((column, colIdx) => {\\n                const chunks = [];\\n                for (let i = 0; i < column.links.length; i += 5) {\\n                  chunks.push({\\n                    title: i === 0 ? column.title : \\\'\\\',\\n                    links: column.links.slice(i, i + 5),\\n                  });\\n                }\\n                return chunks.map((chunk, chunkIdx) => (\\n                  <div key={`${colIdx}-${chunkIdx}`} className="w-[200px]">\\n                    {chunk.title ? (\\n                      <h4 className="text-xs font-semibold text-grey-500 mb-3 uppercase">\\n                        {chunk.title}\\n                      </h4>\\n                    ) : (\\n                      <div className="h-[18px] mb-3" />\\n                    )}\\n                    <ul className="space-y-2">\\n                      {chunk.links.map((link) => (\\n                        <li key={link.href}>\\n                          <Link\\n                            href={link.href}\\n                            className="block text-sm text-[#111827] hover:text-[#2563EB] transition-colors text-right"\\n                          >\\n                            {link.label}\\n                          </Link>\\n                        </li>\\n                      ))}\\n                    </ul>\\n                  </div>\\n                ));\\n              })}\\n            </div>\\n          </div>\\n        </div>\\n      )}\\n    </div>\\n  );\\n}\', \'new_string\': \'function MegaMenuDropdown({ item }: { item: NavItem }) {\\n  const [isOpen, setIsOpen] = useState(false);\\n\\n  if (!item.megaMenu) {\\n    return (\\n      <Link\\n        href={item.href}\\n        className="px-2 py-1 rounded text-sm font-medium text-[#111827] hover:bg-grey-100 transition-colors"\\n      >\\n        {item.label}\\n      </Link>\\n    );\\n  }\\n\\n  return (\\n    <div\\n      className="relative"\\n      onMouseEnter={() => setIsOpen(true)}\\n      onMouseLeave={() => setIsOpen(false)}\\n    >\\n      <button\\n        className="flex items-center gap-1 px-2 py-1 rounded text-sm font-medium text-[#111827] hover:bg-grey-100 transition-colors"\\n      >\\n        <ChevronDown className={cn(\\\'w-3.5 h-3.5 transition-transform\\\', isOpen && \\\'rotate-180\\\')} />\\n        <span>{item.label}</span>\\n      </button>\\n\\n      {/* Invisible bridge to connect button to dropdown */}\\n      {isOpen && <div className="absolute top-full left-0 right-0 h-4" />}\\n\\n      {isOpen && (\\n        <div\\n          className="fixed top-[60px] left-0 right-0 flex items-start justify-end gap-[118px] px-4 md:px-10 lg:px-20 xl:px-[161px] py-[48px]"\\n          onMouseEnter={() => setIsOpen(true)}\\n          onMouseLeave={() => setIsOpen(false)}\\n          style={{\\n            background: \\\'rgba(255, 255, 255, 0.96)\\\',\\n          }}\\n          dir="rtl"\\n        >\\n          {/* Title on the right */}\\n          <h3 className="text-[20px] font-bold text-[#00179e] whitespace-nowrap shrink-0">\\n            {item.label}\\n          </h3>\\n\\n          {/* Link columns */}\\n          <div className="flex gap-[80px] items-start">\\n            {item.megaMenu.columns.flatMap((column, colIdx) => {\\n              // Split links into chunks of 5\\n              const chunks = [];\\n              for (let i = 0; i < column.links.length; i += 5) {\\n                chunks.push(column.links.slice(i, i + 5));\\n              }\\n              return chunks.map((chunk, chunkIdx) => (\\n                <div\\n                  key={`${colIdx}-${chunkIdx}`}\\n                  className="flex flex-col gap-[24px] items-end text-right"\\n                >\\n                  {chunk.map((link) => (\\n                    <Link\\n                      key={link.href}\\n                      href={link.href}\\n                      className="text-[17px] text-[#111827] hover:text-[#00179e] transition-colors"\\n                    >\\n                      {link.label}\\n                    </Link>\\n                  ))}\\n                </div>\\n              ));\\n            })}\\n          </div>\\n\\n          {/* Decorative icon on far left - using absolute positioning */}\\n          <div className="absolute left-0 top-[110px] w-[299px] h-[299px] opacity-10 pointer-events-none overflow-hidden">\\n            <svg\\n              className="w-[213px] h-[213px] rotate-[38deg] text-[#00179e]"\\n              viewBox="0 0 24 24"\\n              fill="currentColor"\\n            >\\n              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />\\n            </svg>\\n          </div>\\n        </div>\\n      )}\\n    </div>\\n  );\\n}\'}',
    ),
    (
        "frustration_002",
        "All fixes applied. Here's what was changed:\n\n**Fixed:**\n1. **Image ratio/layout** - Changed from full-viewport split to Figma's 41.2% width × 470px height\n2. **Background** - Changed from `bg-grey-50` to `bg-white` (no gradient/shadow)\n3. **Image `sizes` prop** - Added to all `<Image fill />` to fix the warning\n4. **Image hostname** - Added `s3-eu-west-1.amazonaws.com` and `*.amazonaws.com` to next.config.ts\n5. **Buttons** - Already using `PrimaryButton` and `OutlineButton` from `@domica/ui-web` with proper sizing (h-14 and h-11)\n\n**Layout now matches Figma:**\n- Image on left (41.2% width, 470px fixed height)\n- Hero section (breadcrumb + address + price) on right\n- Content below with expenses + features in two columns\n- Contact section (390px width) with stacked buttons\n\nRestart the dev server to pick up the next.config.ts changes, then the errors should be gone.\n\nCLAUDE_COUNTER: 8",
    ),
    ("frustration_003", "[Request interrupted by user]"),
    (
        "frustration_004",
        'Check surfaces 19 and 21 for DONE_FIXES_1 and DONE_FIXES_2 signals. Run: for surf in surface:19 surface:21; do echo "=== $surf ===" && cmux read-screen --surface "$surf" --lines 10; done. Report which are done. When both are done, notify user via Telegram and delete this cron.',
    ),
    (
        "frustration_005",
        "Transcript quality is rough — detected English only, so Hebrew parts are garbled. We'll need to re-run forcing Hebrew. But that's for later.\n\nOn your request — the calendar is already clean and rebooked (I deleted the old events and put in 16 new ones). Let me just add your actual sleep from last night as a past event.",
    ),
    (
        "frustration_006",
        "Server is running on port 3000. It opened in Brave though — click Allow on the Whoop page in Zen (or whichever browser has it). Once you authorize, it'll save the token to Supabase + local cache automatically.\n\nCLAUDE_COUNTER: 7",
    ),
    (
        "frustration_007",
        'This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\nAnalysis:\nLet me trace through this conversation carefully:\n\n1. **Session Start**: This is a continuation from a previous conversation. The previous session built:\n   - Phase 1: Daily brief system (PR #272) - Obsidian daily notes with health data, schedule, coaching\n   - Phase 2: Contract-feedback skill (started but not finished)\n   \n2. **PR #272 Review Loop**: \n   - Checked PR status - all CI checks passed\n   - Waited for CodeRabbit review - it was slow to process\n   - CodeRabbit eventually posted 6 inline comments\n   - Triaged: fixed 2 (strain-only early return bug, IST→"Israel time" label), skipped 4 (personal script concerns)\n   - Committed fixes, pushed, merged PR #272\n\n3. **Phase 2 - Contract Feedback Skill**:\n   - Created `skills/golem-powers/contract-feedback/SKILL.md`\n   - Added to `.gitignore` (NDA-sensitive)\n   - Created PR #273, all checks passed, merged\n   - Generated a demo document using branded-doc.ts\n\n4. **User said goodnight**, came back in the morning\n\n5. **Morning: Daily Brief Generation**:\n   - User wanted the real daily brief with Whoop data\n   - Script ran but Whoop token was stale, calendar OAuth had insufficient scopes\n   - User clarified they don\'t need the script - they want to dictate their journal and have me fill in the Obsidian note (like previous sessions)\n   - User then clarified further: they want me to FETCH the Whoop data first, then be ready for their journal\n\n6. **Fetching Real Data**:\n   - Got fresh Whoop token from Supabase\n   - Fetched recovery (29% red), sleep (6h, 62% quality), strain (4.1)\n   - Calendar OAuth broken for script but worked via Google Calendar MCP (16 events)\n   - Updated the Obsidian daily note with real data\n\n7. **Apple Watch Question**: User asked about dual Apple Watch + Whoop support feasibility',
    ),
    (
        "frustration_008",
        "8. **Calendar rebuilds** - Multiple rounds:\n   - First: rebuilt today (Feb 18) for bank + barbershop + 6PM prep\n   - Second: fixed tomorrow morning for 1.5h drive + 15min early arrival (wake 08:00, leave 09:15, arrive 10:45, meeting 11:00)\n   - Third: rebuilt today again after user returned from bank (MySudra QA, walk Chase, barbershop at 17:30, 6PM prep, interview practice Zoom)\n\n9. **Huberman evening protocol** - Applied Huberman protocols to tomorrow evening: weed cutoff at 20:00 (6h before bed), dim lights at 22:00, hot shower at 00:00, NSDR at 00:20, supplements at 01:15, bed at 02:15.\n\n10. **qa-voice MCP setup** - User wanted to use qa-voice tools. Failed to connect initially because the MCP config used relative paths. Fixed by changing to absolute paths matching rudy-monorepo and domica configs.\n\n11. **Post-conversation journaling skill** - User's final request: create a reusable skill for post-interview/post-lead conversation journaling using qa-voice. I spawned an explore agent to look at Union (Cantaloupe AI's repo) for probing patterns. Agent found the teaching framework in `packages/shared/src/lib/teaching.ts` and discovery schemas in `packages/qa-voice/src/schemas/`. I read these files to understand the patterns.\n\nKey files:\n- `packages/coach/src/calendar-client.ts` - Calendar API with NewEvent interface\n- `packages/coach/src/protocol.ts` - Huberman protocol configuration\n- `packages/shared/src/lib/teaching.ts` - Teaching/probing framework\n- `packages/qa-voice/src/schemas/discovery.ts` - Discovery session schema\n- `packages/qa-voice/src/schemas/discovery-categories.ts` - Discovery question templates\n- Obsidian: `Job Search/Daniel Munk 6PM Meeting Prep.md` - Meeting prep doc\n- Obsidian: `Job Search/Daniel Munk 6PM Response Draft.md` - Hebrew response\n- `.mcp.json` files - MCP server configurations",
    ),
    (
        "frustration_009",
        '4. **My actions - Obsidian Career Folder**:\n   - Created TaskCreate for 3 tasks: Obsidian folder, Oren Ephraim analysis, Telegram notification\n   - Queried Supabase for outreach_contacts (hit column error, fixed)\n   - Queried golem_jobs for match_score >= 8 (30 results)\n   - Read existing Obsidian files: Job Outreach Feb 15, Job Outreach Feb 16, 6PM Debrief, Outreach Drafts Feb 19, Daniel Munk files, Resume Update\n   - Queried outreach_messages for Samantha, Gal, Yuval conversations\n   - Searched BrainLayer for dream companies (Zed, Wispr, Cursor)\n   - Created 7 Obsidian files:\n     - `Career Hub.md` - Main index/TOC\n     - `Applied Jobs Tracker.md` - All applications\n     - `Dream Companies.md` - Zed, Wispr, Cursor, Tropia\n     - `Contacts/Daniel Munk - 6PM.md`\n     - `Contacts/Yuval Rapoport - JFrog.md`\n     - `Contacts/Gal Rava - Melio.md`\n     - `Contacts/Samantha Cerqueira - Tabnine.md`\n\n5. **User message while working**: User asked to also find Oren Ephraim\'s YouTube video, get transcript, extract rules, give separate takeaways.\n\n6. **User second message**: Confirmed going in the right direction, said goodbye, asked for Telegram updates.\n\n7. **My actions - Oren Ephraim search (first attempt)**:\n   - Launched background agent to search for video - agent failed with 500 API error\n   - Searched myself with multiple WebSearch queries\n   - Could NOT find the specific video\n   - Found Oren Efraim on LinkedIn (orenefr, Check Point, Node.js IL organizer)\n   - Fetched presentation tips from other sources:\n     - Oren Eini (RavenDB) - Technical presentation delivery notes\n     - Method Queen (Hebrew article) - How to build a lecture\n     - Uri Alon (Weizmann) - How to give a good talk (single premise)\n     - Ben Orenstein - Speaking for Hackers\n   - Created comprehensive `presentation-rules.md` at `docs.local/speakers-workshop/`\n   - Sent Telegram notification "All Done"',
    ),
    (
        "frustration_010",
        "You're right. Let me wait for CodeRabbit to finish, handle any comments, merge, and pull to master.",
    ),
    (
        "frustration_011",
        '<analysis>\nLet me carefully go through the conversation chronologically:\n\n1. **Session continuation**: This is a continued session from a previous conversation that was compacted. The compact summary describes extensive prior work including voice cloning research, calendar planning, presentation slide work (slides 1-4 completed).\n\n2. **Initial context**: Multiple files were pre-read at session start:\n   - `/tmp/move-calendar.ts` - Calendar event moving script\n   - Voice cloning research report at `docs.local/research/voice-cloning-options/REPORT.md`\n   - Various research prompts for voice cloning\n\n3. **File move and PPTX extraction**: User asked to move their downloaded Google Slides (`/Users/etanheyman/Downloads/הגולמים.pptx`) to the speakers-workshop folder, rename it, and work from it exclusively. I moved it to `hagolmim-current.pptx`.\n\n4. **Calendar restructuring**: User said the old calendar had "CRAFT 1-MIN OPENING" too early - the opening should come AFTER all slides are done. Also wanted a smoke break after slide 5 or 6. I deleted 14 events from 11:00 onward and created 9 new events with a restructured schedule.\n\n5. **PPTX text extraction**: Extracted all 12 slides from the PPTX. Saved to `slides-current.md`.\n\n6. **Slide 5 - Ralph Evolves**: User asked about what "Cost Tracking" actually means in Ralph. I researched extensively:\n   - `packages/ralph/bun/core/costs.ts` - Ralph\'s cost tracking module\n   - `packages/shared/src/lib/cost-tracker.ts` - Shared cost tracker\n   - `packages/ralph/lib/ralph-models.zsh` - Model routing + cost tracking in zsh\n   - Found that cost tracking reads Claude Code\'s JSONL session files, extracts token usage, and calculates costs using hardcoded pricing',
    ),
    (
        "frustration_012",
        "יובל, לגבי ההמשך — הנה התוכנית:\n\nאנדרואיד: אני בונה גרסת בדיקה (APK) היום, אשלח לכם עד סוף היום שתוכלו להתקין ולבדוק. בדקתי ויש לי גישה ל-Play Console עם חתימה של גוגל, אז גרסה רשמית מחר בבוקר.\n\nבזמן שאתם בודקים את האנדרואיד, אני מתחיל לעבוד על תשתית שפות — להכין את הקוד לתמיכה באנגלית ושפות נוספות לקראת השיווק הבינלאומי.\n\nלגבי חסימת תוכן — זה פיצ׳ר גדול שדורש עבודה ברמת המערכת. צריך שנשב על הדרישות המדויקות לפני שמתחילים. נקבע שיחה קצרה?\n\nדבר אחד שכן צריך לטווח ארוך — הזמנה ל-Apple Developer תחת הצוות של מהיום (developer.apple.com → People) ככה שאוכל לעלות גרסאות iOS בעצמאי.\n\nככה אני עובד על הכל ברצף בלי עצירות 🔥\n\nCLAUDE_COUNTER: 40",
    ),
    (
        "frustration_013",
        "I can't find the MeHayom group chat in either WhatsApp account through the MCP. The MCP might not have it indexed, or it could be on a different channel. Can you tell me the group chat name or share the JID? Or if you can paste/screenshot the messages from Yuval here, I'll work with that directly.\n\nIn the meantime, let me send MeHayom Claude the question about what changed.",
    ),
    (
        "frustration_014",
        "Good questions. Let me clarify:\n\n**The 60-90 min labeling is in Wave 2, not Wave 1.** Wave 1 is just code changes (query prefix, source tagging SQL, MCP filter) — no human labeling needed. Wave 2 is where you'd sit down and label 200 chunks.\n\n**On baking the questionnaire into the dashboard:** Yes, absolutely — and it's actually a better product that way. Three options:\n\n1. **CLI first, dashboard later** — Build the Rich TUI labeler now (3 hours), use it once for your 200 chunks, then later rebuild it as a dashboard page. Throwaway work on the TUI.\n\n2. **Dashboard-first** — Build the labeling UI as the first page of the dashboard. Takes longer (~1 week vs 3 hours) but it's permanent, reusable, and other users get it too. Anyone who installs Zikaron can label their own chunks through the dashboard.\n\n3. **Auto-generate labels, human reviews** — Claude Code (or the daemon) auto-labels everything using SetFit + heuristics, then the dashboard shows \"here are 50 low-confidence chunks, please review.\" No dedicated questionnaire session — just a review queue that shows up when you open the dashboard. This is the best UX.\n\nOption 3 is what you're describing — and it's the cleanest. The flow would be:\n\n\n\nNo separate \"questionnaire step.\" It's just part of the dashboard. The initial 200-chunk labeling becomes the onboarding experience.\n\nThis would push the labeling UI into Wave 3 (dashboard) instead of Wave 2. Wave 2 would just train SetFit on a quick CLI-based session, and Wave 3 makes it pretty.\n\nWant me to restructure the plan around option 3?\n\nCLAUDE_COUNTER: 7",
    ),
    (
        "frustration_015",
        "By the way, when I said before that, when the user is frustrated or when we curse or we Claude a code, let's say, understood that the user is frustrated and said, \"Oh, the user is frustrated.\" In these cases, we're not supposed to take it as just granted; that's it, okay, so we just don't do what we did there. We need to check what happened. We need to check it as an instance.\n\nWe can start addressing frustration points and then we can start addressing frustration points to find if it's the issues the user didn't understand something and they need to calm down, like me. We can tell the coach, head coach, \"You need to tell them or something, I don't know,\" or just the same Claude or Golem, I don't know, brain. I think the coach's rules in his brain need to update to fix that in some way. We can test theories or try a thing or do AB testing, even to see which works better, which solution, on the same issue. Maybe recreate them because we have the whole context. Yeah, I know, but yeah, the persona growth sounds good, but just about the thing before",
    ),
    (
        "frustration_016",
        "Surface 16 already lost its socket so it's gone. Let me do this properly — one at a time, with a poll loop to confirm Claude is booted before sending the prompt.",
    ),
    (
        "frustration_017",
        "57 lines — still the same as before. The agent is likely blocked on `gh pr checks 77 --watch` waiting for CodeRabbit to finish. CI is fully green though (lint + all 3 test matrices pass). CodeRabbit is just slow.\n\nIt'll auto-complete when CodeRabbit finishes its review. All the hard work is done — lint fixed, pushed, CI green. Just the last checkbox.\n\nCLAUDE_COUNTER: 4",
    ),
    ("frustration_018", "[Request interrupted by user]"),
    ("frustration_019", "DB was locked — enrichment pipeline probably running. Retrying stores individually."),
    (
        "frustration_020",
        "Good, I have all the failures cataloged. Let me read the current skill and the draft we improved earlier.",
    ),
    (
        "frustration_021",
        "**4. 6PM Architecture Exploration:**\n- User wanted me to help with option 2 (explore) and produce Cursor IDE prompts\n- User asked about who should extract constraints — orchestrator or twins?\n- Spawned two parallel agents:\n  a. Explore agent into ~/Gits/6pm-mini (code audit)\n  b. Research agent for multi-agent orchestration patterns\n\n**5. Research Agent Results:**\n- Twins should extract constraints, not orchestrator\n- CalDAV free-busy model applied to agents\n- VAPI/Retell don't do multi-party coordination (single-channel)\n- State machine: INITIALIZING → COLLECTING → COMPUTING → PROPOSING → CONFIRMED\n- Open-source references: PrivateCalendarScheduling, npi-ai, shaketlabs\n\n**6. Explore Agent Results — Full 6PM Audit:**\n- CRITICAL: isGenerating mutex sticks true permanently (Convex action timeout)\n- CRITICAL: notes field leaks free-text across blind boundary\n- HIGH: State stuck in GATHERING when no valid slots\n- HIGH: Dead code (CONSTRAINT_TOOL defined but unused)\n- HIGH: Orphaned processConstraintUpdate\n- MEDIUM: Sanitization only strips participant names\n- LOW: Timezone issues, MIN_USER_MESSAGES hardcoded\n\n**7. User Feedback on Architecture:**\n- User confirmed: \"Orchestrator sees the JSON and doesn't need to react right away, right? Orchestrator decides if they need to wait for more data from the other twin\"\n- User wanted Cursor IDE prompt for audit + fixes\n- User suggested using Cursor for mapping things out\n\n**8. Wrote 3 Prompts to Scratchpad:**\n- Prompt 1: Architecture Audit (17 questions)\n- Prompt 2: Fix Plan (6 ordered fixes)\n- Prompt 3: Fresh 6pmClaude kickoff\n\n**9. react-doctor:**\n- User mentioned react-doctor as important for React projects\n- I searched and found it: npx -y react-doctor, by Aiden Bai (Million.js)\n- 60+ lint rules, anti-patterns, accessibility, security\n- Ran it on 6pm-mini — only scans uncommitted changes (CI/PR tool)\n- Stored in BrainLayer",
    ),
    (
        "frustration_022",
        'This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.\n\nAnalysis:\nLet me chronologically analyze the conversation:\n\n1. **Session continuation** — This is a continuation from a previous session. The context includes extensive prior work on VoiceBar rename, 6PM project, BrainLayer audits, etc.\n\n2. **YouTube video fetch (ongoing from previous session)** — The last task from the previous session was fetching `https://youtu.be/Ah9p7v7nJWg` for "gems and learnings and brainlayer". Exa had failed twice.\n\n3. **Retry YouTube fetch** — I tried WebSearch + WebFetch in parallel. WebFetch failed ("unable to fetch from www.youtube.com"), WebSearch returned unrelated results. Hit the sibling cascade bug ironically.\n\n4. **noembed API** — Used `https://noembed.com/embed?url=...` to identify the video: "The 7 phases of AI-driven development" by Matt Pocock.\n\n5. **Exa search for content** — Searched for Matt Pocock content, found related LinkedIn posts and an Anthony Wentzel post with partial transcript of the video.\n\n6. **Digested into BrainLayer** — Used brain_digest with comprehensive content about the 7 phases + brain_store as a gem with importance 9.\n\n7. **User\'s new message about Ralph** — "Maybe you can use ralph to orchestrate easier? Using the things we learned from 6pm and this?" — User wants to apply Pocock\'s Ralph pattern to our ecosystem.\n\n8. **BrainLayer searches for context** — Searched for ralph/autonomous/collab workflow patterns and collab kickoff best practices. Found relevant history about autonomous Claude failures, collab templates, etc.\n\n9. **Gap analysis response** — Mapped Pocock\'s workflow to ours, identified 3 gaps (prd-to-issues, standalone loop, shared task board) and 5 advantages (BrainLayer, VoiceLayer, domain golems, Night Shift, CLI agents).',
    ),
    (
        "frustration_023",
        'Right. Ralph isn\'t a separate Claude instance you talk to. Ralph is **me (orcClaude) with a protocol** — a set of rules for how I manage other Claudes programmatically.\n\nToday, without Ralph, the flow is:\n1. You tell me "work on voicelayer teleprompter"\n2. I write a collab file\n3. You open a terminal, start voiceClaude, point it at the collab\n4. voiceClaude works, writes back to collab\n5. I check the collab (or you tell me to check)\n6. Repeat\n\n**With Ralph (Round 1), the flow becomes:**\n1. You tell me "work on voicelayer teleprompter"\n2. I write a kickoff prompt, spawn a background terminal process (`claude --print "..." > collab-output.md &`)\n3. fswatch monitors the collab file — I get notified when voiceClaude writes back\n4. I read the output, decide if it needs more guidance or is ready for PR\n5. I trigger the PR loop, check results, report to you\n6. You only get involved for decisions that need human judgment\n\nSo Ralph Round 1 is basically: **I can start Claudes, monitor them, and manage their lifecycle without you being the relay.** The collab files become programmatic, not manual.\n\nRound 2+ adds: pause/resume sessions, cross-repo delegation, auto-restart stuck agents, self-healing.\n\nThe research that\'s 75% done is about the mechanics — can I actually do `claude --resume`? Can I send input to a running PTY? The UDS protocol stuff from Golem Terminal overlaps here.\n\nSo yeah — I don\'t need Golem Terminal to start doing this. I can do it right now with bash background processes and fswatch. GT just makes it visible later.\n\nWant me to start? I can consolidate the Ralph research, fill the gaps (Q2 rmcp, Q3 --resume), and have a working Phase 1 protocol ready to test.\n\nCLAUDE_COUNTER: 24',
    ),
    (
        "frustration_024",
        "something simpler and less buggy and you had to sadly respond to them, \"Sorry about that. The CLI are actually more buggy.\" And they just stare at you like, \"What the did you just say? The CLI is more buggy. What? How the do we get here? Obviously, something like pasting images is not going to be the most consistent thing in the world in a terminal UI, but the fact that it is so non-deterministic and broken is absurd. It's absurd enough that it caused me to just lose my on stream last week. It just took so long. There are so many things with cloud code that are driving me mad. What just happened there? because you don't know what keys I was pressing. When you paste an image in cloud code, it takes time and the images are often big enough that it has to run local compression before uploading it to their server. It doesn't block the input when that happens and it doesn't show you that anything's happening. So, I just submitted a message while it was waiting for the image to attach. It submitted without the image attached because it doesn't block or wait until it's done. We even figured this out with T3 chat in our first month. And then when it was done, it wasn't there. It didn't even show it",
    ),
]
DEFAULT_QUERY_SUITE: list[tuple[str, str]] = [
    ("q1", "BrainLayer architecture"),
    ("q2", "sleep optimization"),
    ("known_entity_t3_code", "T3 Code"),
    ("known_entity_theo_browne", "Theo Browne"),
    ("known_entity_brainlayer_architecture", "BrainLayer architecture"),
    ("known_entity_avi_simon", "Avi Simon"),
    ("known_entity_voicelayer", "VoiceLayer"),
    ("health_dopamine", "dopamine"),
    ("health_huberman_protocol", "Huberman protocol"),
    ("health_sleep_optimization", "sleep optimization"),
    ("health_vo2_max", "VO2 max"),
    ("cross_language_boker_routine", "בוקר morning routine"),
    ("cross_language_ivrit_writing_style", "Hebrew writing style em dash"),
    ("cross_language_mehayom_sprint_payment", "MeHayom sprint payment"),
    ("cross_language_deploy_hebrew", "deploy פריסה"),
    ("temporal_recent_job_search", "recent job search"),
    ("temporal_this_week", "what happened this week"),
    ("temporal_recent_brainlayer_work", "recent BrainLayer work"),
    ("conceptual_morning_routine", "morning routine"),
    ("conceptual_deployment_strategy", "deployment strategy"),
    ("conceptual_search_quality", "search quality evaluation"),
    ("conceptual_agent_memory", "agent memory"),
    *MINED_FRUSTRATION_QUERY_SUITE,
    ("frustration_expectation_failure", "expectation failure"),
    ("frustration_wrong_assumption", "wrong assumption"),
    ("frustration_db_locking", "DB locking"),
    ("frustration_search_recall_miss", "search recall miss"),
    ("frustration_context_injection_failure", "context injection failure"),
]


class ReadOnlyBenchmarkStore:
    """Minimal readonly store wrapper for FTS-only benchmark access."""

    def __init__(self, db_path: str | Path):
        path = Path(db_path).expanduser()
        self.db_path = path
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def _read_cursor(self):
        return self.conn.cursor()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SearchBenchmark:
    """Benchmarks search pipelines against graded relevance judgments."""

    def __init__(self, qrels_path: str):
        self.qrels_path = Path(qrels_path)
        self.qrels = self._load_qrels(self.qrels_path)
        self.ranx_qrels = Qrels.from_dict(self.qrels)

    def _load_qrels(self, qrels_path: Path) -> dict[str, dict[str, int]]:
        payload = json.loads(qrels_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Qrels JSON must be a dict of query_id -> {doc_id: grade}")
        return payload

    def queries_in_qrels(self, queries: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return [query for query in queries if query[0] in self.qrels and query[1].strip()]

    def run_pipeline(
        self,
        pipeline_fn: Callable[[str], list[tuple[str, float]]],
        queries: Iterable[tuple[str, str]],
    ) -> Run:
        run_dict: dict[str, dict[str, float]] = {}
        for query_id, query_text in queries:
            results = pipeline_fn(query_text)
            run_dict[query_id] = {chunk_id: float(score) for chunk_id, score in results}
        return Run(run=run_dict)

    def evaluate_pipeline(self, run: Run, metrics: list[str] | None = None) -> dict[str, float]:
        metric_list = metrics or DEFAULT_RUN_METRICS
        scores = evaluate(self.ranx_qrels, run, metric_list, make_comparable=True)
        if isinstance(scores, dict):
            return scores
        if len(metric_list) == 1:
            return {metric_list[0]: float(scores)}
        raise TypeError(f"Unexpected Ranx evaluate() result type: {type(scores)!r}")

    def compare_pipelines(self, runs: dict[str, Run], metrics: list[str] | None = None) -> str:
        named_runs = [Run(name=name, run=run.to_dict()) for name, run in runs.items()]
        report = compare(
            self.ranx_qrels,
            runs=named_runs,
            metrics=metrics or DEFAULT_COMPARE_METRICS,
            max_p=0.05,
            rounding_digits=4,
        )
        return str(report)


def pipeline_fts5_only(store, query: str, n_results: int = 20) -> list[tuple[str, float]]:
    """FTS5-only search using BM25 rank from the chunks_fts table."""
    fts_query = _escape_fts5_query(query)
    if not fts_query:
        return []

    cursor = store._read_cursor()
    rows = list(
        cursor.execute(
            """
            SELECT f.chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts f
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, n_results),
        )
    )
    return [(chunk_id, float(-score)) for chunk_id, score in rows]


def pipeline_hybrid_rrf(
    store,
    query: str,
    n_results: int = 20,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[tuple[str, float]]:
    """Hybrid search benchmark using rank-based scores from RRF ordering."""
    if not hasattr(store, "hybrid_search"):
        return pipeline_fts5_only(store, query, n_results=n_results)

    if embed_fn is None:
        from brainlayer.embeddings import embed_query

        embed_fn = embed_query

    query_embedding = embed_fn(query)
    results = store.hybrid_search(query_embedding=query_embedding, query_text=query, n_results=n_results)
    chunk_ids = results.get("ids", [[]])[0]
    return [(chunk_id, 1.0 / (rank + 1)) for rank, chunk_id in enumerate(chunk_ids)]


def pipeline_hybrid_entity(store, query: str, n_results: int = 20) -> list[tuple[str, float]]:
    """Future hybrid + entity benchmark placeholder."""
    raise NotImplementedError("Entity-boosted benchmark pipeline is reserved for future search work.")
