# Sample corpus

Plain-text Wikipedia articles for chunking and RAG demos.
The download script calls the MediaWiki API with `prop=extracts&explaintext=1`, which returns cleaned plain text.

## Pages (5)

| File | Article |
| --- | --- |
| `corpus/retrieval-augmented_generation.txt` | Retrieval-augmented generation |
| `corpus/vector_database.txt` | Vector database |
| `corpus/word_embedding.txt` | Word embedding |
| `corpus/large_language_model.txt` | Large language model |
| `corpus/information_retrieval.txt` | Information retrieval |

Refresh:

```bash
source .venv/bin/activate
python scripts/download_wikipedia.py
```

## Attribution

Text from [English Wikipedia](https://en.wikipedia.org/), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each `.txt` file includes title, source URL, and license in a short header.
