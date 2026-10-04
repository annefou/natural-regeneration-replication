# 10 — CiTO Citations: The Unjournal evaluations review the paper (manual, outside the chain)

> Not part of the FORRT chain and not read by `pixi run build-chain-draft` (it only reads
> `01`–`08`). Publish by hand in the Science Live UI, one nanopub per section below.
> **Do not publish yet:** publishing waits for Anne's decision (`DISCOVERY.md`), and
> ideally after discussing with The Unjournal whether they would rather sign these
> links themselves (see `ScienceLive/living-claims-white-paper.md` §7).

**Template:** "Declare citations with CiTO" — `https://w3id.org/np/RA43F9EoOuzF0xoNUnCMNyFsfIqlsuWDdPHCnN0wCdCAw`
(`nanopubs/templates/registry.json`, key `CITATION_CITO`).

**Fields (from `docs/forrt-form-fields.md` § Citation with CiTO):**

| Field label | Field type |
|---|---|
| Identifier for the citing creative work | text input, required |
| List citations | repeatable group, required ≥ 1 |
| ↳ Citation Type | dropdown (CiTO intention; `reviews` is in the list) |
| ↳ DOI or other URL of the cited work | text input |

**Why `reviews`:** it is the CiTO equivalent of the Crossref relation already deposited by
PubPub for the evaluation summary: `is-review-of` → `10.1038/s41586-024-08106-4`
(Crossref API, `https://api.crossref.org/works/10.21428/d28e8e57.5411b150`, retrieved
2026-10-04; Crossref type `peer-review`). `reviews` is neutral; `critiques` would add a judgement
the bibliographic record does not make.

**Who asserts it:** signed by Anne Fouilloux (ORCID 0000-0002-1784-2920) as a citation statement
*about* The Unjournal's work, with Crossref as its evidence. It does not speak for the evaluators.
The Unjournal: ROR `https://ror.org/01yppp702`.

---

## 10a — Evaluation summary and metrics → reviews → paper

### Identifier for the citing creative work (text input, required)

```
https://doi.org/10.21428/d28e8e57.5411b150
```

### List citations (repeatable group, required ≥1)

#### Citation 1

##### Citation Type (dropdown)

```
reviews
```

##### DOI or other URL of the cited work (text input)

```
https://doi.org/10.1038/s41586-024-08106-4
```

---

## 10b — Evaluation 1 → reviews → paper

Crossref: `10.21428/d28e8e57.5411b150/6a67da7e`, type `component`, title "Evaluation 1 of
"Global potential for natural regeneration in deforested tropica…"" (truncated in our check),
author listed as "Evaluator 1" (anonymous). The component record carries no relation itself; the
`is-review-of` link is on the summary, which lists this evaluation under `is-supplemented-by`.

### Identifier for the citing creative work (text input, required)

```
https://doi.org/10.21428/d28e8e57.5411b150/6a67da7e
```

### List citations (repeatable group, required ≥1)

#### Citation 1

##### Citation Type (dropdown)

```
reviews
```

##### DOI or other URL of the cited work (text input)

```
https://doi.org/10.1038/s41586-024-08106-4
```

---

## 10c — Evaluation 2 → reviews → paper

Crossref: `10.21428/d28e8e57.5411b150/640bb340`, type `component`, title "Evaluation 2 of
"Global potential for natural regeneration in deforested tropica…"" (truncated in our check),
signed: one author with an ORCID in the Crossref record. Same note on relations as 10b.

### Identifier for the citing creative work (text input, required)

```
https://doi.org/10.21428/d28e8e57.5411b150/640bb340
```

### List citations (repeatable group, required ≥1)

#### Citation 1

##### Citation Type (dropdown)

```
reviews
```

##### DOI or other URL of the cited work (text input)

```
https://doi.org/10.1038/s41586-024-08106-4
```

---

## Before publishing — checklist

- [ ] Anne decides to publish (and, ideally, has discussed with The Unjournal who should sign).
- [ ] Re-check the three DOIs resolve (`resolve_doi`) and the Crossref `is-review-of` relation is unchanged.
- [ ] Publish 10a, 10b, 10c; record their URIs in `nanopubs/PUBLISHED.md` under a separate
      "Related citations (outside the chain)" heading.
