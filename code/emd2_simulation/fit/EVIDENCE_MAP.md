# EMD2 evidence map: what licenses each typed relationship

Companion to [`FINDINGS.md`](FINDINGS.md), which records what this build
computed, and [`PROVENANCE.md`](PROVENANCE.md), which records where its inputs
came from. This file records **why the domain is assumed**, and which exemplar
carries each typed EMD2→KCC edge. It is prose over public literature, not an
output of the analysis; nothing here is computed.

## Why the domain itself is assumed

EMD2 is a useful Layer-2 domain because microbial metabolism and microbial
products can change internal dose or host response in ways the ten KCCs already
name — **provided the mediator and direction are stated**. Two limbs, pointing
in opposite directions of attribution:

| limb | direction | the microbe acts on |
|---|---|---|
| **1** | agent → microbe → internal dose | the **chemical**: microbial metabolism changes what reaches tissue |
| **2** | microbe → host | the **host**: microbial products change a cancer-relevant response |

Each limb has independent public functional evidence, and they are **different
exemplars**. No single agent is asked to carry both.

## Citation set

| Link | Limb | Best public functional exemplar |
|---|---|---|
| Domain / disposition method | 1 | Zimmermann 2019 *Science* (PMID 30733391) |
| EMD2–KCC1 **home**, + sequence-level tumours | 1 | Roje 2024 *Nature* (PMID 39085612; MTBLS3581; S3/S42) |
| EMD2–KCC2 downstream | 2 | Nougayrède 2006 *Science* (PMID 16902142); Wilson 2019 *Science* (PMID 30765538); Pleguezuelos-Manzano 2020 *Nature* (PMID 32106218); Arthur 2012 *Science* (PMID 22903521) |
| EMD2–KCC6 **home** | 2 | Wu 2009 *Nat Med* (ETBF, PMID 19701206) |
| EMD2–KCC7 + KCC8 downstream | 2 | Hezaveh 2022 *Immunity* (PMID 35139353) — *one programme, two adjacent edges* |
| EMD2–KCC10 downstream | 2 | Wu 2009 ETBF-driven hyperplasia (**not** Roje tumour counts) |
| Domain framing | — | Sepich-Poore 2021 *Science* (PMID 33766858; already in `hkcc.db`) |

Two notes on this table, both corrections to an earlier draft of it:

- **Arthur 2012 belongs under KCC2, not KCC6.** Its load-bearing result is that
  *pks* deletion reduced tumours. It bears on KCC6 only as a **reverse-direction
  caveat**: host inflammation expands genotoxic *pks*+ *E. coli*, and *pks*
  deletion did **not** reduce colitis, so in that model inflammation is not a
  product of the genotoxin. Citing it as the KCC6 exemplar would assert the
  opposite of what it shows. KCC6's home rests on ETBF.
- **KCC7 and KCC8 share one exemplar.** Hezaveh supplies receptor activity
  (AhR) and the immunosuppressive phenotype downstream of the same event. That
  is one mechanism supporting two adjacent edges, which is weaker than two
  independent packages, and the shared row is deliberate. What licenses it in
  either case is the control that excludes the host-intrinsic route: the ligand
  is *Lactobacillus*-dependent, not macrophage tryptophan metabolism.

## What this licenses on the BBN build

**Supported.** EMD2–KCC1: measured transformation product, orthogonal microbial
handles, isolate and community reconstruction, altered disposition, and human
faecal communities converting BBN with donor-level variation. This is
KCC1-related metabolic activation evidence. It is **not** a new electrophilicity
assay of BCPN — nitrosamine-metabolite electrophilicity is prior literature
(α-hydroxylation, DNA alkylation).

**Supported, as sequence-level evidence only.** Microbiome-dependent bladder
tumours (S3, S42), now imported and analysed in `empirical.py`
(`tumour_outcome`). See the caveats below; this scores no KCC.

**Not supported by BBN metabolism or by the model exposure index.** EMD2–KCC6
as a demonstrated home *in these animals*; EMD2–KCC2, KCC7, KCC8 or KCC10 as
demonstrated downstream effects of microbial BCPN.

For BBN specifically, the near misses are informative. Degoricija 2019
(PMID 31779626) documents a time-resolved inflammatory programme in the same
model and laboratory, but has no microbial-conversion arm — so it supports KCC6
for BBN *as an agent*, not EMD2–KCC6. Knezović 2024 (PMID 39000291, same lab)
shows *Myd88* loss reduces invasiveness while *Tlr4* does not — receptor
dependence in the model, but not conversion-linked. BBN/BCPN genotoxicity is
real (Nagao 1977; Airoldi 1994; Toyoda 2013; Fantini 2018) but is never paired
with converting versus non-converting microbiota, so it supports KCC2 for the
agent rather than the EMD2 link.

**KCC6 therefore remains a conditional home on the BBN exemplar.** It is not an
unsupported invention of the domain: ETBF and colitis-associated *pks*+
*E. coli* already occupy that home.

## The gap the tumour tables do not close

Isolate-level conversion is causally tied to BCPN. Antibiotic depletion is
causally tied to tumours. **The full isolate → tumour chain is not in that
paper.** Antibiotics remove the converting organism *and* everything else, so
S3/S42 cannot separate loss of conversion from any other consequence of
depleting the microbiota; the arms that would separate them — the reconstructed
consortia — are metabolomic at days to three weeks, not 20-week tumour studies.

## Must not be done

- Counting EMD2 as an extra positive on top of the KCC scores.
- Treating 16S community shifts as EMD2 evidence.
- Treating tumour incidence as KCC10.
- Treating a urinary or urothelial exposure index as genotoxicity or host
  response. In this build the index is exactly proportional to cumulative
  urinary excretion, so it is not even independent of urine.

## Next evidentiary step

S3 and S42 were the only Roje tables adding a **new kind** of evidence without
changing the KCC definitions, and they are now imported. Tables S6, S7 and S13
remain unimported: they are further KCC1 disposition data of a kind already
represented, and S7 in particular is the table behind the intraluminal-ordering
discrepancy recorded in `FINDINGS.md`. Importing them would extend an existing
category rather than open a new one.
