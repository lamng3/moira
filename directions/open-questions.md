# Open questions

Research notes. These are not product decisions, and they are not part of the docs site.

## Reverse questions, question keys and answer keys

A stored answer begins “An anatomic region is a part of the body that has a specific location and function, but does not have well-defined compartmental boundaries.” The next question turns that definition around: “What is a part of the body that has a specific location and function, but does not have well-defined compartmental boundaries?”

That question missed the hot hash. It went through concept retrieval and `ollama:llama3.1`, the long path in the chat (Thought 33.1s, then another working card). A definition asked back should name the concept without another full model call. The per-answer checksum in `src/agentoi/memory/signature.py` now does that for this literal case: three-word shingles of the question overlap the stored answer, the last concept on that path is named, and retrieval is skipped.

Lookup today is question-keyed. `normalize_question` in `src/agentoi/memory/path.py` folds the question, and `HotCache` matches that string. The Harness compares the new question with cached questions, not with stored answers. The concept trie in `src/agentoi/memory/trie.py` is a prefix of concept ids. The answer sits on the terminal node and is not a search key.

The same answer also repeats the prompt heading “RELEVANT CONCEPT CLUSTERS” from `make_prompt_for_query` in `src/agentoi/algorithms/graph/concept_graph.py`. The stored value is a restatement plus prompt text, not the concept name, so it is a poor key.

An answer trie is only a partial fit. A prefix trie helps when the question is a literal span of a stored answer. It does not help when the definition is paraphrased, and a polluted answer poisons the trie.

The order that runs today:

1. Exact question hash, as the hot cache does now.
2. A checksum of three-word shingles on each remembered answer. Overlap of at least 0.75, leading the next answer by 0.15, names the last concept on that path. Fewer than 4 shingles continues. The index is rebuilt from the concept trie and the long-term patterns. It is not a new cache file.
3. The existing Harness, still comparing questions.
4. The answer model.

Paraphrase matching stays open. A definition that does not share those three-word windows still misses the checksum.

## Answer convergence

“What are the organ systems in the mouse?” and “list the organ systems in the mouse” are one request with a different frame. Each wording had its own hot entry, so the chat showed two answers.

Questions that keep the same content words now share one hot answer. Frame words such as list, what, are, and the do not open a second entry. A trailing plural, such as system and systems, shares that key too. When several stored answers share that key, the fuller one is returned. A paraphrase that changes the content words, such as “which body systems does a mouse have”, still stays open. The Harness can replay those after the content-word key misses, when it judges the intent to be the same.

## Bloom and cuckoo in front of the checksum

The per-answer checksum names the concept. A bloom filter or a cuckoo filter only answers whether a shingle might have been seen. That is useful once the index is larger than the 32 hot answers: a question whose shingles are all absent can skip the scan. A hit still has to rank the per-answer checksums, because the filter cannot say which answer matched.

A cuckoo filter can delete a shingle when LFU evicts an answer. A plain bloom filter cannot. Counting blooms can delete, at more bits per shingle. False positives only cost a scan of the checksums. Paraphrases still miss, because both filters and the checksum need the same three-word windows. No filter is implemented in this change.
