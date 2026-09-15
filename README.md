# Contextual-Gate-Transformer
This is a project on contextual-gate extractive offline text summarizer with semantic relevance. 


An extractive summarization pipeline built to test one specific question: does a learned sequential gating layer help or hurt summarization quality as documents get longer — and if it hurts, is that due to the architecture itself or simply insufficient training exposure to long sequences?

This project follows on from a short-document (XSum) extractive summarizer where a similar gated component showed no quality advantage over simple baselines (TF-IDF, semantic + MMR). Rather than assume the result would transfer, this project tests it directly on genuinely long documents (GovReport, up to several thousand words), and includes a second architecture designed to specifically address a diagnosed cause of failure.

Project structure
long_doc_project/
├── utils/
│   ├── sentence_splitter.py        # NLTK-based sentence splitting
│   └── text_loader.py              # loads .txt / .pdf documents
├── contextual_gate_summarizer.py   # flat gate architecture + candidate-unit pipeline
├── hierarchical_gate_summarizer.py # local-gate + global-attention architecture
├── baseline_tfidf.py               # sparse lexical baseline
├── semantic_mmr_baseline.py        # MiniLM semantic + MMR baseline
├── prepare_training_data.py        # builds long-document training set
├── prepare_evaluation_data.py      # builds long-document held-out test set
├── train_ranker.py                 # trains the flat ContextualGateRanker
├── train_hierarchical_ranker.py    # trains the HierarchicalGateRanker
├── benchmark.py                    # runs all configurations, computes ROUGE
├── requirements.txt
└── data/ models/ output/           # generated at runtime, not version-controlled
Architectures
1. Flat Contextual Gate (contextual_gate_summarizer.py)

A single GRU-style gated recurrent layer (ContextualGateLayer) processes the entire document's sentence embeddings sequentially, producing one importance score per sentence (ContextualGateRanker). Combined with semantic relevance, TF-IDF keyword score, and sentence position for the final ranking.

Explicitly not a Mamba/selective-state-space implementation — no input-dependent state matrices, no selective scan. It is a plain gated recurrence, and is named accordingly throughout the code.

2. Hierarchical Local-Gate + Global-Attention (hierarchical_gate_summarizer.py)

Designed after diagnosing why the flat gate struggles on long documents (see Findings below). Splits the document into fixed-size chunks (CHUNK_SIZE = 30 candidates); the same ContextualGateLayer runs independently within each chunk (bounding what any single hidden state has to carry), then a small multi-head self-attention layer runs across chunk-level pooled representations to capture document-level relevance cheaply. Each sentence's final score combines its local in-chunk representation with its chunk's globally-attended representation.

Both rankers implement the same forward(embeddings) -> scores interface, so either can be passed directly into the shared extractive_summarize() pipeline without modification.

Setup
bash
pip install -r requirements.txt --break-system-packages
python -m nltk.downloader punkt punkt_tab
Run order

All commands below assume you are inside the long_doc_project/ folder (cd long_doc_project first if you're not).

bash
# 1. Build the long-document train/test sets (GovReport, ~150 train / 30 test)
python prepare_training_data.py
python prepare_evaluation_data.py

# 2. Train the flat gate
python train_ranker.py

# 3. Train the hierarchical gate (uses the same data, independent model)
python train_hierarchical_ranker.py

# 4. Run the full benchmark — automatically includes the hierarchical
#    model if its checkpoint exists, skips it gracefully if not
python benchmark.py

benchmark.py produces three CSVs in output/:

long_doc_benchmark_detailed.csv — per-document results, every configuration
long_doc_benchmark_overall.csv — averaged across all documents
long_doc_benchmark_by_length.csv — the key result, ROUGE broken down by document-length band (medium / long / very_long)
Findings so far
The flat gate degrades ROUGE at every document length, and the degradation grows as documents get longer (a ~0.017 ROUGE-1 gap on medium documents widening to ~0.032 on very-long documents, gate enabled vs. disabled, same pipeline otherwise).
Isolating cause (architecture vs. training exposure): capping inference to the same sequence length used in training (200 candidates) shrank the gap by roughly 20% but did not close it — indicating the degradation is primarily architectural (a plain gated recurrence loses effectiveness on long sequences even within its trained length range), with a smaller secondary contribution from limited exposure to long sequences during training.
Hierarchical local-gate + global-attention was designed to directly target the architectural cause identified in (2), by bounding the sequence length any single gate instance has to handle. (Result pending — add here once benchmark.py has been run with both checkpoints trained.)
Known limitations
Training uses ~150 documents — enough to see a trend, not enough for strong statistical claims.
The flat gate's training truncates documents to 200 candidate units; 96/150 training documents (64%) hit this cap in the run this project was built from. The hierarchical model uses a looser 600-candidate cap since it does not need the same protection.
Only one training run per architecture so far — no repeated seeds to confirm result stability.
Evaluation set is 30 held-out documents.
Timing/RAM numbers can be noisy if the benchmark is run on a loaded machine — rerun on an idle system before reporting exact figures.
Relationship to the short-document (XSum) project

This project intentionally mirrors the structure of an earlier short-document extractive summarizer, using the same candidate-unit preprocessing, the same ROUGE library (rouge_score, unified across all scripts), and comparable baselines — so that findings here can be directly compared against short-text results (where the flat gate similarly showed no advantage over TF-IDF/semantic baselines).