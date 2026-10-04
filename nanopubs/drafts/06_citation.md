# 06 — CiTO Citation

> Run the pre-flight checklist in `docs/forrt-form-fields.md` § Pre-flight checklist before drafting.

**Description:** *"Declare citations between papers or other works, using Citation Typing Ontology"*

## Field-by-field draft

<!-- field: work -->
### Identifier for the citing creative work (text input, required)

URI of the Outcome published in step 05. Pull from `nanopubs/PUBLISHED.md`.

```

```

### List citations (repeatable group, required ≥1)

#### Citation 1 — back to the original paper

##### Citation Type (dropdown)

Choose based on the Outcome's validation status:

- Validated → `confirms`
- PartiallySupported → `qualifies`
- Contradicted → `disputes`

For question-rooted chains where there is no original paper to confirm/dispute, use `usesMethodIn` or `citesAsAuthority` for the methodology paper(s).

Write the chosen type in the block below (a vocabulary label such as `cites as authority`, or `citesAsAuthority`). `build-chain-draft` uses it as written; leave the block empty to have the type derived from the Outcome's validation status, which is right for paper-rooted chains only.

> **Note:** `replicates` is NOT in the Science Live dropdown (despite existing in upstream CiTO). When citing a notebook/tutorial that was directly reused, use **`credits`** instead.

```

```

##### DOI or other URL of the cited work (text input)

```
https://doi.org/10.1038/s41586-024-08106-4
```

#### Additional citations (optional)

If the Outcome cites methods papers, related replications, or upstream tools, add them here.

One line per further citation, in this exact form (each becomes a pre-filled row):

- Type: usesMethodIn → URL: https://doi.org/10.21428/d28e8e57.5411b150
- Type: citesAsDataSource → URL: https://doi.org/10.5281/zenodo.23138775
- Type: citesAsDataSource → URL: https://doi.org/10.5281/zenodo.7428804
- Type: citesAsDataSource → URL: https://data.globalforestwatch.org/content/pantropical-tree-plantation-expansion-2000-2012/about
- Type: citesAsDataSource → URL: https://colombia.mapbiomas.org/
- Type: citesAsDataSource → URL: https://doi.org/10.1126/science.1244693

  (The Unjournal evaluation summary of the paper: its evaluators' suggested robustness checks — predictors from
  before the outcome period, forward prediction, label omission, other algorithms — shaped steps 3, diagnostics
  and the MapBiomas experiments. Agreed with Anne 2026-10-04.)

  (citesAsDataSource: the Outcome's evidence comes from the derived dataset; the Outcome template has no
  dataset field, so the typed link lives here. Anne, 2026-10-04.)

  (Further citesAsDataSource rows: the authors' published maps (reproduction), Fagan et al. 2022 regrowth labels
  (Global Forest Watch), MapBiomas Colombia Collection 3 (independent labels), Hansen Global Forest Change.)

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 06.

This completes the six-step FORRT chain. Optional next layers:

- **Research Software** (`drafts/07_research_software.md`) — if the repo *produces* a reusable software artefact.
- **Research Synthesis** (`drafts/08_synthesis.md`) — if this chain is one of several testing facets of a shared property.
