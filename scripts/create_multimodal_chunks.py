import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
base_chunks_path = ROOT_DIR / "data" / "chunks.json"

with open(base_chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} base text chunks.")

# Explicit Multimodal Chunks for Tables, Figures, Diagrams, Charts
multimodal_assets = [
    {
        "id": "chunk_table_1_2_p41",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 41,
        "metadata": {
            "type": "table",
            "asset_name": "Table 1-2",
            "chapter": 1,
            "section": "ai_exposure_occupations"
        },
        "text": """[Table 1-2 | Page 41 | Source: Eloundou et al. (2023)]
Table 1-2. Occupations with the highest exposure to AI as annotated by humans.
Alpha (α) refers to exposure to AI models directly, whereas beta (β) and zeta (ζ) refer to exposures to AI-powered software.
Highest exposed occupations:
- Interpreters and translators: close to 100% exposure
- Survey researchers: high exposure
- Poets, lyricists and creative writers: high exposure
- Animal scientists: high exposure
- Public relations specialists: high exposure
- Writers and authors: close to 100% exposure
- Tax preparers: close to 100% exposure
- Web designers: close to 100% exposure
Occupations with 0% exposure to AI:
- Cooks, stonemasons, athletes: 0% exposure (no tasks reduced by 50% by AI)."""
    },
    {
        "id": "chunk_table_1_3_p42",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 42,
        "metadata": {
            "type": "table",
            "asset_name": "Table 1-3",
            "chapter": 1,
            "section": "generative_ai_use_cases"
        },
        "text": """[Table 1-3 | Page 42]
Table 1-3. Common generative AI use cases across consumer and enterprise applications.
Categorized into eight groups based on analysis of 205 open source GitHub repos (>=500 stars) and 50 enterprise company interviews:
1. Coding (code generation, code refactoring, code explanation, code translation)
2. Content creation (writing, blogging, marketing copy, social media posts)
3. Customer support (chatbots, ticket routing, sentiment analysis)
4. Knowledge management (enterprise search, documentation Q&A, internal wikis)
5. Education (personalized tutoring, quiz generation, course creation)
6. Healthcare (medical summarization, clinical notes, patient communication)
7. Gaming & Entertainment (NPC dialogue, procedural generation, story branching)
8. Legal & Finance (contract review, financial report analysis, compliance checking)."""
    },
    {
        "id": "chunk_table_2_1_p76",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 76,
        "metadata": {
            "type": "table",
            "asset_name": "Table 2-1",
            "chapter": 2,
            "section": "common_crawl_languages"
        },
        "text": """[Table 2-1 | Page 76 | Source: Lai et al. (2023)]
Table 2-1. The most common languages in Common Crawl, a popular dataset for training LLMs.
Columns: Language | Code | Population (Million) | CC size (%) | Category
1. English | en | 1,452 | 45.8786% | High (H)
2. Russian | ru | 258 | 5.9692% | High (H)
3. German | de | 134 | 5.8811% | High (H)
4. Chinese | zh | 1,118 | 4.8747% | High (H)
5. Japanese | jp | 125 | 4.7884% | High (H)
6. French | fr | 274 | 4.7254% | High (H)
7. Spanish | es | 548 | 4.4690% | High (H)
8. Italian | it | 68 | 2.5712% | High (H)
9. Dutch | nl | 30 | 2.0585% | High (H)
10. Polish | pl | 45 | 1.6636% | High (H)
11. Portuguese | pt | 257 | 1.1505% | High (H)
12. Vietnamese | vi | 85 | 1.0299% | High (H)
Note: English accounts for almost half (45.88%) of Common Crawl, about 8x more prevalent than the second most common language (Russian at 5.97%)."""
    },
    {
        "id": "chunk_table_2_3_p81",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 81,
        "metadata": {
            "type": "table",
            "asset_name": "Table 2-3",
            "chapter": 2,
            "section": "clip_open_clip_benchmark"
        },
        "text": """[Table 2-3 | Page 81]
Table 2-3. Open CLIP and CLIP's performance on different image datasets.
Comparison between OpenAI CLIP ViT-B/32 and Open CLIP ViT-B/32 accuracy (%):
- ImageNet: CLIP = 63.2%, Open CLIP = 62.9%
- ImageNet v2: CLIP = N/A (-), Open CLIP = 62.6%
- Birdsnap: CLIP = 37.8%, Open CLIP = 46.0% (Open CLIP +8.2% higher)
- Country211: CLIP = 17.8%, Open CLIP = 14.8%
- Oxford 102 Category Flower: CLIP = 66.7%, Open CLIP = 66.0%
- German Traffic Sign Recognition Benchmark: CLIP = 32.2%, Open CLIP = 42.0% (Open CLIP +9.8% higher)
- Stanford Cars: CLIP = 59.4%, Open CLIP = 79.5% (Open CLIP +20.1% higher)"""
    },
    {
        "id": "chunk_table_2_4_p88",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 88,
        "metadata": {
            "type": "table",
            "asset_name": "Table 2-4",
            "chapter": 2,
            "section": "llama_dimensions"
        },
        "text": """[Table 2-4 | Page 88 | Source: Touvron et al. (2023), Dubey et al. (2024)]
Table 2-4. The dimension values of different Llama models.
Columns: Model | # transformer blocks | Model dim | Feedforward dim | Vocab size | Context length
- Llama 2-7B: 32 blocks | Model dim: 4,096 | Feedforward dim: 11,008 | Vocab size: 32K | Context length: 4K
- Llama 2-13B: 40 blocks | Model dim: 5,120 | Feedforward dim: 13,824 | Vocab size: 32K | Context length: 4K
- Llama 2-70B: 80 blocks | Model dim: 8,192 | Feedforward dim: 22,016 | Vocab size: 32K | Context length: 4K
- Llama 3-7B: 32 blocks | Model dim: 4,096 | Feedforward dim: 14,336 | Vocab size: 128K | Context length: 128K
- Llama 3-70B: 80 blocks | Model dim: 8,192 | Feedforward dim: 28,672 | Vocab size: 128K | Context length: 128K
- Llama 3-405B: 126 blocks | Model dim: 16,384 | Feedforward dim: 53,248 | Vocab size: 128K | Context length: 128K"""
    },
    {
        "id": "chunk_table_2_5_p94",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 94,
        "metadata": {
            "type": "table",
            "asset_name": "Table 2-5",
            "chapter": 2,
            "section": "parameters_vs_tokens"
        },
        "text": """[Table 2-5 | Page 94 | Compute Optimal Training Tokens]
Table 2-5. Examples of the number of training tokens for models with different numbers of parameters.
- MT-NLG 530B (Smith et al., 2022): Model Size = 530 billion parameters, Training tokens = 270 billion tokens (undertrained according to Chinchilla scaling law).
- Chinchilla 70B (Hoffmann et al., 2022): Model Size = 70 billion parameters, Training tokens = 1.4 trillion tokens (Chinchilla compute-optimal: 20 tokens per parameter).
- Llama 1 65B: 65 billion parameters trained on 1.4 trillion tokens.
- Llama 3 8B: 8 billion parameters trained on 15 trillion tokens (overtrained by ~90x relative to compute-optimal to minimize inference cost)."""
    },
    {
        "id": "chunk_fig_1_1_p27",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 27,
        "metadata": {
            "type": "diagram",
            "asset_name": "Figure 1-1",
            "chapter": 1,
            "section": "tokenization"
        },
        "text": """[Figure 1-1 | Page 27 | Sơ đồ Kiến trúc / Minh họa]
Figure 1-1. An example of how GPT-4 tokenizes a phrase.
The diagram visualizes the string tokenization process in OpenAI GPT-4.
Input string: "You can't do this!"
Tokenized segments:
1. Token "You" (leading space and word)
2. Token " can" (subword morpheme)
3. Token "'t" (negative contraction subword)
4. Token " do" (word)
5. Token " this" (word)
6. Token "!" (punctuation token)
Key visual insight: In GPT-4 tokenization, the contraction "can't" is broken into two distinct subword tokens: "can" and "'t"."""
    },
    {
        "id": "chunk_fig_1_2_p29",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 29,
        "metadata": {
            "type": "diagram",
            "asset_name": "Figure 1-2",
            "chapter": 1,
            "section": "autoregressive_vs_masked"
        },
        "text": """[Figure 1-2 | Page 29 | Sơ đồ Kiến trúc]
Figure 1-2. Autoregressive language model and masked language model architecture.
Visual comparison between two language modeling architectures:
1. Left panel - Autoregressive (Causal) Language Model (e.g. GPT):
   Given preceding tokens (x1, x2, ..., x_t-1), the model predicts the next token x_t in a strictly unidirectional left-to-right fashion.
2. Right panel - Masked Language Model (e.g. BERT):
   Random tokens are replaced by a [MASK] symbol within the sequence, and the model uses bidirectional context (both left and right tokens) to predict the masked token."""
    },
    {
        "id": "chunk_fig_1_5_p37",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 37,
        "metadata": {
            "type": "chart",
            "asset_name": "Figure 1-5",
            "chapter": 1,
            "section": "sp500_ai_mentions"
        },
        "text": """[Figure 1-5 | Page 37 | Biểu đồ Xu hướng]
Figure 1-5. The number of S&P 500 companies that mention AI in their earnings calls over time.
Type: Bar chart / Time series.
X-axis: Quarters from 2013 to 2024.
Y-axis: Number of S&P 500 companies mentioning 'AI', 'artificial intelligence', or 'machine learning'.
Key data points:
- From 2013 to 2022: Steady baseline of ~50 to 100 companies mentioning AI.
- Q1 2023 (after ChatGPT release): Sharp exponential spike to over 150 companies.
- By Q1 2024: Nearly 200 S&P 500 companies (almost 40% of the entire index) mentioned AI in their quarterly earnings calls."""
    },
    {
        "id": "chunk_fig_1_7_p43",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 43,
        "metadata": {
            "type": "chart",
            "asset_name": "Figure 1-7",
            "chapter": 1,
            "section": "use_case_distribution"
        },
        "text": """[Figure 1-7 | Page 43 | Biểu đồ Phân bố / Pie Chart]
Figure 1-7. Distribution of use cases in the 205 open source repositories on GitHub.
Type: Distribution chart analyzing 205 repositories with >= 500 stars.
Breakdown by percentage:
1. Coding & Software Development: 28% (highest share, rank 1)
2. Personal Assistants & Chatbots: 21%
3. Knowledge Management & Search (RAG): 17%
4. Content Creation & Writing: 14%
5. Education & Tutoring: 8%
6. Gaming & Entertainment: 5%
7. Others / Miscellaneous: 7%
Conclusion: Coding tools represent the single largest share among open-source generative AI projects."""
    },
    {
        "id": "chunk_fig_1_8_p43",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 43,
        "metadata": {
            "type": "chart",
            "asset_name": "Figure 1-8",
            "chapter": 1,
            "section": "internal_vs_external_deployment"
        },
        "text": """[Figure 1-8 | Page 43 | Biểu đồ Khảo sát Doanh nghiệp]
Figure 1-8. Companies are more willing to deploy internal-facing applications than external-facing applications.
Source: a16z Growth Report (2024).
Chart comparison:
- Internal-facing applications (e.g. employee knowledge search, code assistant, internal documentation): Over 70% of enterprise companies surveyed have deployed or are actively running pilots.
- External customer-facing applications (e.g. customer service bots, public generative agents): Less than 30% of companies have deployed to production due to hallucinations, compliance risks, brand safety, and data privacy concerns."""
    },
    {
        "id": "chunk_fig_1_9_p45",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 45,
        "metadata": {
            "type": "chart",
            "asset_name": "Figure 1-9",
            "chapter": 1,
            "section": "developer_productivity"
        },
        "text": """[Figure 1-9 | Page 45 | Biểu đồ Năng suất Lập trình viên]
Figure 1-9. AI can help developers be significantly more productive, especially for simple tasks and less experienced developers.
Source: McKinsey study on developer productivity with generative AI.
Key metrics:
- Documentation generation: ~2x productivity improvement (50% reduction in time).
- Code generation & code refactoring: 25% to 50% productivity improvement.
- Complex architecture / novel algorithms: Minimal productivity improvement (<10%).
- Experience gap: Junior developers experienced higher relative productivity gains (+35-45%) compared to Senior developers (+15-20%)."""
    },
    {
        "id": "chunk_fig_1_11_p58",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 58,
        "metadata": {
            "type": "chart",
            "asset_name": "Figure 1-11",
            "chapter": 1,
            "section": "reasoning_cost_over_time"
        },
        "text": """[Figure 1-11 | Page 58 | Biểu đồ Chi phí Suy luận]
Figure 1-11. The cost of AI reasoning rapidly drops over time.
Source: Katrina Ligett & AI index data.
X-axis: Timeline from 2021 to 2024.
Y-axis: Cost per million tokens on standardized benchmark (MMLU performance equivalent).
Observations:
- In early 2022, achieving MMLU ~70% cost roughly $30 per million tokens (GPT-3/text-davinci-002).
- By late 2023, models achieving higher accuracy (GPT-3.5-turbo, Claude Instant) dropped below $1.50 per million tokens.
- By mid 2024 (gpt-4o-mini, Claude 3 Haiku), cost dropped to under $0.20 per million tokens, representing a >100x cost reduction in two years while benchmark performance increased."""
    },
    {
        "id": "chunk_fig_1_14_p62",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 62,
        "metadata": {
            "type": "diagram",
            "asset_name": "Figure 1-14",
            "chapter": 1,
            "section": "ai_engineering_stack"
        },
        "text": """[Figure 1-14 | Page 62 | Sơ đồ Kiến trúc 3 Lớp]
Figure 1-14. Three layers of the AI engineering stack.
Diagram showing the architectural hierarchy:
1. Top layer - Application Development: Prompt engineering, RAG (retrieval-augmented generation), agent orchestration, evaluations, user interface.
2. Middle layer - Model Development: Pre-training, fine-tuning (SFT), alignment (RLHF, DPO), model quantization, pruning, distillation.
3. Bottom layer - Infrastructure: Compute clusters (GPUs, TPUs), distributed training frameworks, vector databases, inference serving engines (vLLM, TensorRT-LLM)."""
    },
    {
        "id": "chunk_fig_2_19_p122",
        "source_id": "chip-huyen-ai-engineering-ch1-2",
        "source": "Chip Huyen - AI Engineering_ Building Applications with Foundation Models-O'Reilly Media (2025).pdf",
        "page": 122,
        "metadata": {
            "type": "chart",
            "asset_name": "Figure 2-19",
            "chapter": 2,
            "section": "test_time_compute_curve"
        },
        "text": """[Figure 2-19 | Page 122 | Biểu đồ Test Time Compute]
Figure 2-19. Performance curve as the number of sampled outputs increases in test time compute.
Source: OpenAI experiment on test-time compute with verifiers.
X-axis: Number of candidate outputs sampled (from 1 to 1,000 outputs).
Y-axis: Task accuracy / Pass rate.
Key trend:
- Sampling more outputs initially leads to steady performance gains (from 1 to ~400 samples).
- Peak performance occurs at approximately 400 outputs.
- Beyond 400 outputs, accuracy begins to decline due to adversarial samples fooling the reward model / verifier."""
    }
]

# Append multimodal assets
all_chunks = chunks + multimodal_assets
print(f"Total chunks with multimodal assets: {len(all_chunks)} ({len(chunks)} text + {len(multimodal_assets)} visual/table chunks)")

out_path = ROOT_DIR / "data" / "chunks_multimodal_advanced.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Saved to {out_path}")
