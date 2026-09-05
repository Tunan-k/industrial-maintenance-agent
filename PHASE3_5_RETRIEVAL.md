# Phase 3.5: real local semantic retrieval validation

The Retriever architecture is unchanged. This validation uses the existing
KnowledgeChunk JSONL, a trained multilingual E5-small model, and real temporary
local Qdrant. It does not call an LLM or change diagnostic inference/upload APIs.

## Local configuration

Export an existing trained model directory using `EMBEDDING_MODEL_PATH`. It takes
precedence over the legacy `EMBEDDING_MODEL` variable. No model is downloaded by
the validation script or tests. For multilingual E5-small use these prefixes:

```powershell
$env:EMBEDDING_MODEL_PATH = 'F:\git-demo-file\industrial-maintenance-agent\artifacts\embedding\multilingual-e5-small'
$env:EMBEDDING_REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
$env:EMBEDDING_QUERY_PREFIX = 'query: '
$env:EMBEDDING_DOCUMENT_PREFIX = 'passage: '
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TEMP = 'F:\git-demo-file\industrial-maintenance-agent\artifacts\embedding\tmp'
$env:TMP = $env:TEMP
```

Validation model: `intfloat/multilingual-e5-small`, weight SHA-256:
`1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477`.
The model/config files were obtained from public mirrors because the local
network could not resolve huggingface.co. ModelScope's weight hash matches the
Hugging Face mirror listing. Only safetensors are loaded; remote code is disabled.

## Reproduce

Run from the repository root:

```powershell
.\venv\Scripts\python.exe -B -m scripts.validate_semantic_retrieval --top-k 3 --output reports/phase3_5_retrieval.json
$env:RUN_SEMANTIC_TESTS = '1'
.\venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -q
git diff --stat
git status
```

The real-model tests are opt-in so ordinary offline unit tests do not require
large local weights. When enabled, a missing model fails; there is no fake
embedding fallback. The saved report contains all three queries, complete
Top-K Evidence with score/source/page, model configuration, corpus coverage,
and the non-matching-equipment query result.

The validation run stores its model and temporary files on F: because the C:
temporary directory ran out of space during the first attempt. These local
assets are ignored by Git. Adjust the exported paths for another workstation.

## Interpretation boundaries

- Filters are applied in Qdrant before Top-K.
- Scores are cosine similarity, not confidence or verified fault probabilities.
- Every hit must match an original chunk's text/metadata and drilling_pump type.
- PDF page numbers are retained. Missing PPTX/HTML page provenance stays null.
- The E5 model has a 512-token input limit. Oversized original chunks are listed
  explicitly in `excluded_chunks`; the original corpus is not rewritten or
  silently truncated. Coverage must be read alongside retrieval results.
- Topic terms in returned text are a modest semantic sanity check. A related
  paper or project slide is not automatically a maintenance procedure or a
  supported causal explanation. Human review of the returned excerpts remains
  necessary, especially for the requested causes/inspection/remediation intent.

Reference: [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small).

## Observed result (2026-09-05)

Real-model execution indexed 46/63 chunks. All 17 exclusions were PDF chunks
above the model token limit. The original 63 chunks were preserved.
All three queries returned the same three PPTX chunks in the same order:

| Query | Top 1 | Top 2 | Top 3 |
| --- | ---: | ---: | ---: |
| 钻井泵吸入阀故障原因和检查方法 | 0.908986 | 0.882074 | 0.869885 |
| 压力异常可能原因 | 0.863402 | 0.836080 | 0.823185 |
| 设备振动异常处理措施 | 0.861673 | 0.842974 | 0.829434 |

- Top 1: `chk_0eac7d37996da0f982f7`: equipment damage and sensor arrangement.
  Topic-related, but not a specific suction-valve inspection or repair procedure.
- Top 2: `chk_fed75cb96f7b6e9a1f00`: dataset preparation/model training workflow.
  It does not answer the maintenance-action intent.
- Top 3: `chk_c127fd4722c47b8806f6`: body is only "汇 报 提 纲".
  This is a low-information hit and should not be treated as supporting evidence.

All three sources are `钻井泵.pptx`, equipment_type is `drilling_pump`, and
page/slide are null in the original metadata. No page numbers were invented.
A compressor filter returned an empty list.

Conclusion: real encoding/search/filtering/provenance plumbing is verified.
Reliable cause/inspection/remediation evidence retrieval is NOT established.
Passing schema/filter/topic-sanity tests must not be presented as passing a
semantic relevance benchmark. Scores above 0.8 here do not establish usefulness.
See `reports/phase3_5_retrieval.json` for complete original excerpts and coverage.

Validation: with `RUN_SEMANTIC_TESTS=1`, full pytest completed with **67 passed, 1 warning in 100.08 seconds**. The warning is the existing diagnosis date-parser warning. No semantic tests were skipped.
