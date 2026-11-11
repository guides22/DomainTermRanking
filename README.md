# CheckYourSmile

*Academic NLP exploration of domain-specific term ranking.*

Documentation in preparation. The `notebooks/` folder contains experimental implementations of Hybrid2, SIFRank and SPLADE; `results/` contains preserved experiment outputs.


# Experimental corpora

The original text corpora are not distributed in this repository.

| File used during experiments | Documents | Domain | Notes |
| --- | ---: | --- | --- |
| `ml_en.txt` | 50,000 | Machine learning | Scientific titles and abstracts |
| `ml_en_junior.txt` | 500 | Machine learning | First 500 records of `ml_en.txt` |
| `oral_surgery_en.txt` | 594 | Oral surgery | Scientific titles and abstracts |

Each line represents a title and abstract. The exact mapping between the original corpus filenames and the result folder labels (`checkyoursmile`, `nouveau_corpus`) is not yet established. Source URLs and corpus construction details were not retained.

To rerun experiments, obtain the required corpora separately and update the local input paths in the notebooks.
