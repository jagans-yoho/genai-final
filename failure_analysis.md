# Task 4: RAG Failure Analysis and Groundedness Report

## Evaluation Summary
- **Answerable Questions (6/6)**: Retrieved top 3 relevant chunks with accurate page citations.
- **Unanswerable Questions (4/4)**: Returned strictly `"I don't know."` without hallucination.

## Failure Analysis Matrix
| Question | Observed Failure Type | Root Cause | Implemented Fix |
| "What protocols are supported for auto-backup?" | Retrieval Failure (Prior to fix) | Chunk overlap was 0, protocol names were sliced across chunk boundaries. | Set `chunk_overlap=50` with `RecursiveCharacterTextSplitter`. |
| "Can I install on Nintendo Switch?" | Generation Failure (Hallucination risk) | Standard prompt tried to speculate compatibility. | Added strict system instruction: *"Answer ONLY using context. If not found, reply strictly 'I don't know.'"* |