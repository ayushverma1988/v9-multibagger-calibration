# Restore registered catalyst sources after working-core retention — 18 November 2026

A working-core restore can legitimately omit raw PDFs while retaining their reviewed records, text hashes, issuer identities and original source clocks. The seed installer previously skipped existing records even when their original bytes were missing. A hash-only document index could also choose a different URL for an identical document.

The worker now fetches only the approved, checked-in origins for those existing reviews. It verifies original bytes, extracted text, the exact URL/symbol/ISIN and saved review fields before making an original available. Recovery receipts distinguish the first revalidation from later refreshes and impose a prospective minimum availability time. A same-hash mirror cannot replace the reviewed provenance. Conflicting originals or retained text remain unavailable; the worker does not change their values or rewrite a historical forecast.

Raw bytes remain in the existing transient raw directory; durable restart storage retains only the small recovery receipts and established source evidence. The existing public-source budgets, schedule, fixed heads, price eligibility, registration gates, and promotion rules are unchanged.
