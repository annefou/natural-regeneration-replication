# 01 — Quote-with-comment (paper-rooted chains)

> Run the pre-flight checklist in `docs/forrt-form-fields.md` § Pre-flight checklist before drafting.
>
> If this is a question-rooted chain, use `01_pico.md` or `01_pcc.md` instead — see `docs/chain-decision-tree.md`.
>
> **After choosing the chain shape, delete the two step-1 alternates you aren't using.** Once you've decided this chain is paper-rooted and keep `01_quote.md`, run:
> ```bash
> rm nanopubs/drafts/01_pico.md nanopubs/drafts/01_pcc.md
> ```

**Form heading:** *"Annotate a paper quotation — Annotating a paper quotation with personal interpretation"*

## Field-by-field draft

<!-- field: paper -->
### Cited DOI (text input, required)

Format: starts with `10.` — bare DOI, **NOT** `https://doi.org/...` form.

```
10.1038/s41586-024-08106-4
```

### Quote mode (radio button)

- [x] **Quote whole text (less than 500 characters)**
- [ ] Quote start/end *(use this if the quote exceeds 500 chars)*

<!-- field: quotation -->
### The exact quotation from the paper (max. 500 characters) (textarea, required)

Verbatim from the paper PDF in `paper/`. Character-for-character. ≤ 500 chars in whole-text mode.

Source: main text (Results), PDF p. 133 (`paper/williams-2024.pdf`). Chosen by Anne on 2026-10-03: the replication (Colombia) tests the model/accuracy claim, not the pantropical 215 Mha total. Copied from the PDF text; identical in the Europe PMC JATS XML.

Verification: `verify_quote` (forrt-research MCP) was not available in the analysis session, so this quote is **unverified by that tool**. Mechanical check against the `pdftotext` extraction: matches after whitespace normalisation only (no character altered). Re-run `verify_quote(pdf_path="paper/williams-2024.pdf", quotation=...)` before publishing.

```
The validation accuracy estimate of 87.9% was based on an independent set of 4.87 million random points (equally stratified with respect to the two levels of the dependent variable; further details are provided in Supplementary Information 1 and Extended Data Fig. 4).
```

Character count: 268 / 500.

<!-- field: quotation-end -->
### End of quotation (optional - use when quoting beginning and end of a longer passage, max. 500 characters) (textarea, optional)

Only when quoting the beginning *and* end of a longer passage — set the mode above to
**Quote start/end**, put the opening phrase under the previous heading and the closing
phrase here. Leave empty for a single short quote.

```

```

<!-- field: comment -->
### Our interpretation and explanation of why this quotation is relevant (max. 800 characters) (textarea, required)

Why this quote matters and what the replication tests. Connect the paper's claim to the work this repo does. Don't repeat the quote.

<!-- PROPOSED by Claude on 2026-10-04 for Anne to edit or replace (it is her comment; <= 500 characters, the live form limit). Alternative quotes are listed in 00_paper_summary.md. -->

```
This accuracy underpins the paper's maps and area estimates, and The Unjournal's evaluators (doi:10.21428/d28e8e57.5411b150) doubted it: predictors overlap the outcome period, and random validation may overstate performance. We test it in Colombia with independent code and open data: reproducing the published figures, then validating with spatial blocks, a model trained elsewhere, independent MapBiomas labels and a forward test (2000-2012 to 2012-2024).
```

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 01.
