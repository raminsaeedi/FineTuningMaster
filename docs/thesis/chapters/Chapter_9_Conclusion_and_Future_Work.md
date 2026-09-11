# Chapter 9 — Conclusion and Future Work

This chapter closes the thesis. It restates what the study set out to do, summarises what the evidence establishes and at what strength, states plainly what remains open, and describes the next steps that follow directly from the open items rather than from speculation. It is written so that a reader who has followed Chapters 6 to 8 can see exactly which claims are carried forward and which are withheld.

## 9.1 What the thesis did

The thesis asked whether retrieval-augmented generation, QLoRA fine-tuning, or their combination improves a language model's ability to turn a short dashboard brief into a structured, machine-readable dashboard-design recommendation. To answer that, it built a frozen and lineage-annotated dataset for a task that no public benchmark covers, implemented four system conditions that share a prompt, a parser, an evaluation path and a held-out split, executed thirty-two complete runs across four base models spanning roughly 1.49B to 27B parameters and two model families, and analysed the outcome with paired item-level statistics computed from stored per-item records.

The design held constant everything that was not the intervention. The same 274 held-out items, verified by digest in all thirty-two run manifests, were used for every condition. Every model has a complete A–D matrix at the primary seed, so the four-way method contrast is available at all four scales. Condition D reused the adapter produced by condition C for the same model, dataset, training configuration and seed rather than training a second one, so that the incremental effect of retrieval after adaptation could be isolated from a second stochastic training trajectory.

## 9.2 What the evidence establishes

**Output-contract compliance is acquired by the base model between 1.7B and 8B parameters, and below that threshold it dominates everything else.** Qwen3-8B and Qwen3.8-27B satisfy the contract on all 274 items in both base-model conditions. Qwen3-1.7B satisfies it on under 1.5 % of items, failing on one recurring type error, and OLMo-2-1B-Instruct recovers a complete object on 30–33 % of items because almost every response exceeds the 512-token generation budget before it can close.

**The pre-specified accuracy metric conflated three different failures, and separating them changes the conclusions — including one sign.** Re-scoring the stored raw outputs at three levels of strictness separates the chart decision from output-contract compliance and from the generation-length budget. Measured at the text level, the effect of fine-tuning on the chart decision is +0.36 to +10.58 percentage points, positive in all eight blocks and significant in seven. Measured by the pipeline metric the same contrast ranges from −7.66 points at 8B to +85.04 points at 1.7B. At 8B the pipeline metric reports that fine-tuning hurt the chart decision; the raw text shows that it helped by 4.38 points and separately cost thirty-six responses to truncation.

**Retrieval alone significantly lowers chart agreement at both larger Qwen3 scales.** The effect is −9.49 points at 8B and −8.76 points at 27B, both from code-state homogeneous contrasts and both identical at all three measurement levels because those conditions have no format or truncation losses. At the two small models retrieval is small and positive or neutral. Where retrieval clearly helped, it helped a weak model produce more usable structure — a 15–17 point gain in JSON recovery for OLMo, replicated across three seeds — rather than a better chart choice.

**That negative effect has one traceable cause, and it replicates.** The retrieved corpus states a preference for donut over pie. At 27B this raised pie-to-donut substitutions from 5 to 28 of the 36 pie items; at 8B it raised them from zero to 28. In both cases the substitution accounts for the entire measured drop. Two independently sized models followed one sentence of guidance, and the reference-agreement metric counted every instance as an error. Whether donut or pie is the better recommendation for those items cannot be decided by this metric.

**Retrieval after adaptation is negative in almost every case.** The D − C contrast at the text level is negative in eleven of twelve blocks and significantly negative in eight, with effects of −3.7 to −8.4 points for Qwen3-1.7B, −18.98 for Qwen3-8B and −27.4 to −36.5 for OLMo. At 8B it additionally costs 12.04 points of output validity on a code-state homogeneous contrast. The single exception is the 27B model, where D − C is +0.73 points and not significant. Combining the two interventions cannot be assumed to combine their benefits.

**The marginal value of the interventions falls with capacity, and one of them reverses sign.** At 1.7B, fine-tuning supplies both the output contract and about ten points of chart agreement. At 8B and 27B the contract is already satisfied and fine-tuning supplies about four points, together with the highest macro-F1 observed in the study at 27B (0.762 against 0.445 for prompt-only). Retrieval is mildly positive at 1.7B and significantly negative at both larger scales.

**The same fine-tuning recipe is not equally safe at the two larger scales.** The 8B and 27B base models agree to within one percentage point on every prompt-only and retrieval quantity. They diverge on what the adapter costs: at 27B it adds agreement and macro-F1 with no loss of validity, while at 8B it adds text-level agreement but loses 13.87 points of contract satisfaction and lowers macro-F1 from 0.558 to 0.530, because its losses fall on the small chart classes. "Fine-tuning with this recipe is safe at large scale" is not supported; it was safe at one of the two large scales tested.

**Adaptation removes a behaviour that no reported metric captures.** Every fine-tuned condition of every model offers alternatives on exactly zero items, while Qwen3-8B offers them on all 274 items in both base-model conditions and Qwen3.8-27B on 266 and 267. The adapters reproduced the frozen targets faithfully, and the targets predominantly contain an empty alternatives list.

**The Qwen3 conclusion transfers to an independently developed family for the chart decision but not for usable output.** OLMo-2-1B-Instruct matches Qwen3-1.7B on the chart decision in the prompt-only condition (86.1–87.6 % against 86.9–87.2 %) and gains from fine-tuning at the text level. It diverges sharply on delivering that decision inside the contract and the budget: fine-tuning takes its JSON recovery from about 30 % to 2.9 %. Because architecture, tokenizer, pretraining corpus, post-training recipe and native context length differ simultaneously, no single factor can be named as the cause; the comparison bounds the generality of the main result rather than explaining it.

**Where systems produce usable output they are stable under the tested paraphrase, and none of them asks for missing information.** Paraphrase consistency reaches 100 % for both Qwen3-8B base-model conditions, 96.7–100 % for the 27B model and 81–99 % for the current-state fine-tuned 1.7B conditions, with accuracy losses of at most 1.5 points. The clarification rate on under-specified briefs never exceeds 3.65 % and is exactly zero in every fine-tuned condition of every model.

**Seed variability is small for untrained conditions and non-trivial for trained ones.** Standard deviations across three seeds are 0.2–1.1 points for prompt-only and retrieval conditions and up to 4.7 points for the trained conditions in the weaker family. Only the two smaller models have three-seed coverage.

## 9.3 What the thesis does not establish

The following are stated as explicitly as the findings above, because a reader is entitled to know the boundary.

The thesis makes **no claim about the design quality or usefulness of any generated recommendation**. Layout, styling, accessibility, interaction design, rationale quality and overall usefulness were never judged by a person. The human-evaluation study is fully specified, its infrastructure is implemented, and no ratings exist.

It makes **no claim that any system selects charts that are effective for real users**. Every chart number is agreement with the project's frozen reference. The independent effectiveness layer, which would score predictions against sets of charts shown effective in published human-subject studies, is specified and not implemented.

It makes **no claim about retrieval quality**. Retrieval relevance was not measured, because no independent relevance judgements exist. The grounding numbers measure lexical overlap between rationales and retrieved passages, and ten of the sixteen retrieval conditions have claim coverage too low to interpret.

It makes **no claim that QLoRA is superior to other adaptation algorithms**. One QLoRA configuration was used; no algorithm comparison was run.

It makes **no strong causal claim for the fine-tuning contrasts**. Seven of forty contrasts are code-state homogeneous, and none of them is a C − A contrast. Where a pair is not homogeneous, a significant result shows that two runs differ, not that the method is the sole reason.

It makes **no claim about variance at 8B or 27B**, where only one seed was executed.

Finally, it makes **no claim about behaviour outside the observed distribution**: a held-out split of 274 items dominated by comparison/bar tasks, two perturbation families, one guideline corpus of 41 chunks, and one output contract.

## 9.4 Contributions, restated at the strength the evidence supports

1. **A frozen, lineage-annotated dataset** for the dashboard-brief-to-recommendation task, with field-level evidence classes, generated material labelled as generated, a byte-identical held-out test across freeze steps, and published manifests and hashes. Contribution: the dataset exists, is auditable and is reproducible from its manifest. Not claimed: that its dashboard-level fields are expert gold.

2. **A configuration-driven implementation of four controlled conditions** with enforced adapter provenance, verified for all eight condition-D runs. Contribution: the C-to-D pairing is documented and checked. Not claimed: that all runs share one code state — they do not, and this is reported per contrast.

3. **A layered evaluation protocol** that separates parse success, contract validity, completeness, chart agreement, robustness and grounding, and that states the evidence class of each. Contribution: the separation is implemented and every reported number carries its class.

4. **A complete A/B/C/D comparison at four model scales across two families**, with paired item-level statistics, Holm-corrected exact tests and bootstrap intervals computed from stored per-item records, and with code-state comparability reported per contrast rather than assumed.

5. **A three-level decomposition of the headline chart metric** that separates chart-decision failures from output-contract failures and generation-budget failures, and that materially changes the interpretation of every method effect in the study, including one sign reversal at 8B. Contribution: the decomposition is reproducible from the artifacts by one script.

6. **A traceable and replicated mechanism for a negative retrieval result.** The −9 point retrieval effect at two independent scales is attributed to one identified sentence in the retrieval corpus, with the substitution counted item by item. Contribution: a negative result explained rather than merely reported.

7. **Two documented measurement artifacts** that would have propagated silently into the conclusions: a strict validator with a construction ceiling of 0 % verified against the 274 reference records, and a generation budget below the length of the training target, isolated cleanly at 8B where the same base model truncates zero responses without an adapter and sixty-nine with an adapter and retrieval.

8. **A transparent record of the unmeasured layers**, listed above and carried into the future work below.

## 9.5 Future work

The next steps follow from the gaps rather than from a wish list, and they are ordered by how much they would change what the thesis can claim.

**Run the human evaluation.** This is the single most valuable remaining step, because it is the only one that would let the work say anything about design quality. The study is fully specified: 40 frozen items, four methods, three independent ratings per output, six raters, six 1–5 dimensions, anonymised outputs with method and model hidden, Krippendorff's ordinal α per dimension, Friedman and Holm-corrected Wilcoxon tests, and a derived human chart-acceptability outcome analysed with Cochran's Q and exact McNemar tests. Nothing needs to be redesigned. Three findings in particular are unresolvable without it: whether the donut-for-pie substitutions that retrieval induced at 8B and 27B are improvements or errors, whether the stacked-bar-to-bar collapse matters to a reader, and whether losing the alternatives list makes the fine-tuned systems less useful in practice.

**Implement the independent chart-effectiveness scorer.** The gold table exists in the repository and the design is documented. Scoring predictions as set membership against charts shown effective for the corresponding task and data shape, and reporting covered accuracy and coverage rate separately, would replace agreement with the project's own reference by a criterion the project did not construct. It would also settle the donut question independently of human raters, if the effectiveness table covers those items.

**Re-run inference with a generation budget matched to the target length.** The 512-token ceiling produced the most severe apparent failures in the study, and the 8B block shows unambiguously that they are an artifact of that ceiling. Repeating the OLMo conditions and the fine-tuned Qwen3 conditions with a budget sized from the actual distribution of target lengths would establish what those models achieve when the ceiling is not binding. This requires no retraining.

**Re-run the affected conditions under one pinned code state.** Nineteen of the forty contrasts involve a run whose git commit was not recorded. Re-executing inference for those conditions from one commit, with pinned model revisions, would convert most of the reported contrasts from descriptive to controlled — in particular the C − A contrasts, which carry the fine-tuning conclusion and are homogeneous nowhere. Only inference is required; the existing adapters can be reused, so the cost is a fraction of the original experiment.

**Align or retire the strict response schema.** The strict contract should either accept the source-grounded encoding keys that the reference carries, in which case it becomes an informative measure, or be removed from the reported metric set.

**Add alternatives to the training targets.** The alternatives list collapsed to empty because the targets contain empty lists. Supplying defensible alternatives in the supervision would move the fine-tuned systems closer to the behaviour a design assistant should have, and would restore a capability that the base models at 8B and 27B already exhibit and that adaptation currently destroys.

**Extend the seed coverage at the larger scales.** Three seeds at 8B and 27B would let the scale trend rest on the same evidential footing as the small-model comparison, and would show whether the divergence between the 8B and 27B fine-tuning outcomes is stable or a single-run accident.

Two further directions are worth naming but are further from the current evidence. A matched comparison between QLoRA and one alternative adaptation method — plain LoRA is the closest, since it isolates quantization — would answer whether the choice of adaptation algorithm matters for this task; the trainer already supports the alternatives. And a comparison between constrained decoding and fine-tuning as routes to output-contract compliance would test directly whether the largest measured effect in this study can be obtained without any training at all.

## 9.6 Closing remark

The study set out to compare prompting, retrieval, fine-tuning and their combination on a structured dashboard-design task. It found that at the scales where such a system would plausibly be deployed, most of what an intervention appears to buy is the ability to emit a well-formed record, and that the underlying design decision changes far less than a single accuracy number suggests. It found that retrieval, the intervention most often assumed to be a safe addition, significantly lowered agreement at both larger scales for a reason that turned out to be one sentence in the retrieval corpus. And it found that the metric measuring the design decision was entangled with two implementation choices, which was detectable only because the raw outputs had been kept. None of these is the result the study expected to report, and all three are more useful than the result it expected. What the study cannot yet say is whether any of the recommendations it produced would help a person build a better dashboard. Answering that requires the human evaluation, and until it is run the honest conclusion is bounded accordingly.

## References Used in Chapter 9

Generated from `docs/thesis/references.bib` by `experiments/scripts/build_chapter_reference_lists.py`. The BibTeX key of each entry is given in brackets.

This chapter cites no external sources.
