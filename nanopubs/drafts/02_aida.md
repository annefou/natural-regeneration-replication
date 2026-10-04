# 02 — AIDA Sentence

> Run the pre-flight checklist in `docs/forrt-form-fields.md` § Pre-flight checklist before drafting.

**Form heading:** *"AIDA Sentence — Make structured scientific claims following the AIDA model"*

## Field-by-field draft

<!-- field: aida -->
### AIDA sentence (text input, required)

Atomic, Independent, Declarative, Absolute. One empirical finding. Must end with a full stop.

> _If your draft AIDA contains "and" linking two distinct findings, split into two AIDA nanopubs._

```
A random-forest model of natural forest regeneration in tropical and subtropical forest biomes, trained on natural regrowth that occurred between 2000 and 2016 and applied at 30 m resolution, classifies independent validation points, equally stratified between regenerated and not regenerated, with 87.9 % accuracy.
```

<!-- field: topic -->
### Select related topics/tags (search/select, optional)

Predefined topic vocabulary — list the labels you intend to pick from the dropdown.

```
natural regeneration (Q11442890)
ensemble learning (Q245652)
tropical forest (Q1048194)
```

<!-- field: project -->
### Relates to this nanopublication (search/select, required)

URI of the nanopub the AIDA derives from.

- For paper-rooted chains: the Quote-with-comment URI (from step 01).
- For question-rooted chains: the PICO or PCC URI (from step 01).

Pull the URI from `nanopubs/PUBLISHED.md`.

```

```

<!-- field: dataset -->
### Supported by datasets (text input, optional)

DOIs/URLs of datasets that ground the AIDA claim.

- DOI 1: https://doi.org/10.5281/zenodo.7428804 (the authors' published maps, CC BY 4.0)

<!-- field: publication -->
### Supported by other publications (text input, optional)

DOIs/URLs of publications that support the AIDA claim — e.g. peer-reviewed methods papers, or the original paper if not already cited via the Quote.

> **Known platform bug (2026-04-26):** if both *Supported by datasets* AND *Supported by other publications* are populated and publishing fails, fall back to publishing this AIDA via Nanodash. The URI namespace becomes `https://w3id.org/np/...` (still valid and citable).

*(skip: the original paper is already cited by the Quote, and filling both this field and "Supported by datasets" can make publishing fail — known platform bug above.)*

## Publication note

After publishing, paste the resulting URI into `nanopubs/PUBLISHED.md` step 02.
