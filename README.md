# DomainTermRanking

Academic exploration of domain-specific term importance and keyphrase ranking, developed in the context of the CheckYourSmile project.

## Notebooks

- [`hybrid2.ipynb`](notebooks/hybrid2.ipynb) — TopicRank, YAKE, and RAKE score fusion.
- [`sifrank.ipynb`](notebooks/sifrank.ipynb) — SIFRank/SIFRank+ using ELMo and SIF weighting.
- [`splade.ipynb`](notebooks/splade.ipynb) — SPLADE-v3-based keyphrase ranking inspired by ERU-KG.

Each notebook contains an English-language methodology description, the reference publications, a small self-contained example corpus, an optional user-supplied corpus mode, and its dependency notes. The notebooks are research artifacts; full end-to-end execution has not been verified in a clean environment.

## Data

Original experiments used three UTF-8 scientific-text corpora, with one document per line (typically `Title.. Abstract`):

| Filename | Domain | Documents |
|---|---|---:|
| `ml_en.txt` | Machine learning | 50,000 |
| `ml_en_junior.txt` | Machine learning subset (first 500 documents) | 500 |
| `oral_surgery_en.txt` | Oral surgery | 594 |

**The original TXT corpora are not distributed.** To reproduce an experiment with your own file, place it in a local `data/` directory and change `CORPUS_SOURCE` and `CORPUS_PATH` in the relevant notebook. `data/` is excluded by `.gitignore`.

## Dependencies and reproducibility

Install dependencies in a **separate environment for each notebook** when needed; the notebook's first sections explain its setup and external assets.

| Notebook | Key dependencies | Environment considerations |
|---|---|---|
| Hybrid2 | `pke`, `yake`, `rake-nltk`, `nltk`, `spacy` | English/French spaCy models and NLTK datasets; no GPU required |
| SIFRank | Python **3.6.13**, AllenNLP **0.8.4**, PyTorch, ELMo, Stanford CoreNLP **3.9.2**, Java, spaCy, scikit-learn, pandas | Original legacy setup; pretrained ELMo files and Stanford CoreNLP installation must be supplied separately |
| SPLADE | `sentence-transformers` with `SparseEncoder`, `huggingface_hub`, `spacy`, `numpy`, `pandas`, `matplotlib`, `tqdm` | Downloads `naver/splade-v3`; consult its model page for access requirements |

Exact dependency versions have not been pinned for Hybrid2 and SPLADE, and compatibility was not validated with a clean install. These notebook setup cells are instructions, not a tested cross-platform environment specification.

## Results and research context

Original ranking figures and CSV results are preserved in [`results/`](results/). They show extracted term rankings and scores, **not a quantitative accuracy benchmark**. An academic literature review and poster, coauthored with the other group members and supervisors, are in [`docs/`](docs/).

This repository showcases the author's work on **Hybrid2, SIFRank/SIFRank+, and SPLADE** within a collaborative academic project. The full platform CheckYourSmile and other team members' implementations are not included.

## References and authorship

See the individual notebooks for method publications. The joint article and poster credit Wassim Chenouf, Ange Colletto, Guillaume Andrea Desaphy, and Brice Zozime, supervised by Olfa Mechi and Nadia Yassine-Diab. This repository does not imply sole authorship of those collaborative documents.

<!-- README intentionally concise; expand once notebook dependencies and experimental claims are verified. -->
