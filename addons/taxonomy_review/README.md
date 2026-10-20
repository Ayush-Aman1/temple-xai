# Add-on 3 — Taxonomy review (1–2 h, Member 1)

`checklist.csv` lists every machine-compiled definition with each source (URL + the ≤25-word quote it was written from).

## Steps
1. For each row read the quote (and the page if needed). Mark `ok_(y/n)`. If a definition or visual cue is wrong or too strong, write the corrected text in `correction`; put a better citation in `better_source`.
2. Create `templexai/knowledge/taxonomy_reviewed.yaml` containing **only the entries you changed or approved**, e.g.
   ```yaml
   components:
     gopuram:
       definition: "…your corrected definition…"
       visual_cues: ["…"]
       sources:
         - {page: "…", url: "https://…", retrieved: "2026-..-..", quote: "≤ 25 words, verbatim from the source"}
   ```
   Any entry present in that file overrides the machine one and is marked `HUMAN_REVIEWED`. (To approve an entry without changes list it with its current text.)
3. If you add new sources, save the page text as `data/knowledge_sources/<Title>.json` (same keys as the existing snapshots: title, url, retrieved, licence, text) so the validator can check the quotes: `python -m templexai.knowledge.validate`.
4. Run `claude --model sonnet "$(cat prompts/03_HUMAN_ADDONS.md)"`. No retraining happens.
