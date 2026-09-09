# Active plan — 2026-09-09, after completion audit

The recent W&B window contains 36 finished runs (20 core reproductions, 8 corrected direct, 4 graph ablations, 4 grouping runs) and 7 crashed direct attempts with completed replacements. All 20 core held-out endpoints and retained checkpoint hashes are verified. Do not relaunch these completed series.

- Current evidence: [RESULTS_REVIEW_2026-09-09.md](RESULTS_REVIEW_2026-09-09.md).
- Acceptance queue: [PAPER_COMPLETION_CHECKLIST.md](PAPER_COMPLETION_CHECKLIST.md), 6/12 MUST DONE; 5 PARTIAL and 1 TODO remain.
- Completed meeting-series specification: [NEXT_ACTIONS_AFTER_TRANSFER_2026-09-07.md](NEXT_ACTIONS_AFTER_TRANSFER_2026-09-07.md).
- Historical temp=.5 direct review: [TRANSFER_REVIEW_2026-09-07.md](TRANSFER_REVIEW_2026-09-07.md).
- Historical CRP review: [REPLICATION_REVIEW_2026-09-07.md](REPLICATION_REVIEW_2026-09-07.md).
- Prior direct-screen specification, retained unchanged: [NEXT_TESTS_2026-09-07.md](NEXT_TESTS_2026-09-07.md).

Core CoSpRo test mean delta is +1.41 Avg/+3.82 WGA pp vs SimCLR and +0.74/+2.16 vs raw KL; seed-wise superiority remains mixed. Corrected direct SpLiCE at temperature .05 passes its two-seed validation screen: +2.67/+4.50 vs matched SimCLR and +1.21/+2.68 vs raw distillation, both metrics positive on both seeds. The historical .5 failure and failed CRP replication gate remain recorded. CRP/semantic graph ablation supports WGA but has mixed Avg; semantic concept grouping has negative mean deltas.

Next work is evidence and writing: merge the historical/recent registry, recover exact cluster source and dated/probe provenance for the existing lock, package graph/null/chance evidence and deterministic qualitative panels, and complete the remaining mechanism/qualitative manuscript sections. Test plus all additional series, per-seed/group tables and revised claims were integrated into the manuscript and paper_results.json on 2026-09-09; M08 is DONE. M11 test closure is DONE; M10 provenance remains PARTIAL. The separate FP32-equivalence AMP verification report and qualitative panels have not been verified locally.

Core test has been opened and its no-test-driven-tuning rule remains active. Meeting/direct/grouping results stay validation-only and are excluded by the core lock. A positive direct screen is preliminary evidence, not authorization for a new confirmation series or reuse of the opened test for tuning. No new training is scheduled by this audit.
