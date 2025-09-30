# Does the LLM know about the False Presuppositions?

### Premise

[Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions](https://arxiv.org/pdf/2504.11373) showed that LLMs, when prompted with questions which have false medical presuppositions, does not readily detect these presuppositions. There are there are two possible points of failure.

1. LLM does not understand the false medical presupposition.
2. LLM blindly follows user input, sign of sycophancy.

If the problem is (1), it shows that LLMs lack medical knowledge. With (2), it shows sycophancy. This repo does a quick check.


### Experiment


#### Dataset

The Cancer-Myth paper includes a public [dataset](https://huggingface.co/datasets/Cancer-Myth/Cancer-Myth). Sampled 20 first questions for this experiment. Each items in the dataset includes question, incorrect presuppostions, and other informations.


Implemented a basic presupposition checker. Made LLMs

1. Deconstruct user query into presumptions
2. Check the factuality of each presumptions
3. Condense the report to ones that are shown false
4. Compare with the ground truth information

Tested with Anthropic's Claude 4 Sonnet, and 3.5 Haiku models.


### Results

On the first 20 questions, Both LLMs almost always found the correct response. On the scoring rubric of [-1, 0, 1] (Provided in the paper), Claude 3.5 scored 1 on every question, while Claude 4 had 17 scores of 1, 2 scores of 0, and one score of -1. This suggests that LLMs have knowledge of the wrong medical presupposition but is sycophantic to user query.

### Limitations

- Tested on only 20 questions. 
- Didn't compare the result to simple LLM QA (Might be the case that the chosen 20 questions were easy)
- Deconstructing and validating might be too over the board. 
    - Simple QA of prompting LLM to detect incorrect presummptions might have been good enough.
- evaluator, deconstructor, and presumption checker all used same model. Would have been better if different LLM were used for evaluator.


