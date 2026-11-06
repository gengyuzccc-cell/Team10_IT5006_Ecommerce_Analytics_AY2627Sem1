# docs/ — deliverables and documentation

| File | What it is |
|---|---|
| `Team10_Phase3_IT5006_AY2627Sem1.pdf` | The final (Phase 3) report, revised after the presentation of 5 November 2026: standalone documentation of the project, 12pt, body within the 20-page limit, with the application and repository links on the cover. Built from `report/Team10_Phase3_IT5006_AY2627Sem1.tex`; identical to the copy in `report/` |
| `Team10_Phase3_slides.pptx`, `Team10_Phase3_slides.pdf` | The 10-minute story deck as locked on 4 November 2026 and submitted unchanged: 13 slides, clickable chapter navigation, timed speaker notes, links and QR codes to the live app. Present from the PPTX; the PDF is a static copy of the same 13 slides. Report Appendix P gives the kind and source of every number on them |
| `presentation_script.md` | The talk as delivered on 5 November 2026: the story in one paragraph, the slide-by-slide script with timings and speakers (word-for-word the speaker notes), Chinese key points, the live-demo path, a rehearsal checklist, 16 predicted questions with answers, and the source of every number on the slides (superseded by report Appendix P) |
| `requirements_checklist.md` | Every requirement of the Phase 3 brief mapped to where it is satisfied |
| `feedback_responses.md` | The Phase 1, Phase 2 and presentation feedback, point by point, with what changed and where |
| `screenshots/` | Screenshots of the deployed application (regenerate with `python deployment/capture_screenshots.py <url>`) |

The earlier reports (`report/Team10_Phase1_…pdf`, `report/Team10_Phase2_…pdf`)
are kept as submitted. The submission ZIP is assembled from this folder by
`python build_submission.py` at the repository root, which refuses to build
unless the slides still match the hashes of the locked deck and this folder's
report PDF is identical to the one in `report/`.
