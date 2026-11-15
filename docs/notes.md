
## 1. Exact Identifiers

Query: "What does 4.2 say?"

Observation: The model retrieved the sentence about Section 4.2, but its cosine similarity score was only 0.318 in the initial experiment.

Conclusion: Semantic search can retrieve the relevant sentence, but exact identifiers may receive relatively low similarity scores, especially when the query is short.

## 2. Numbers

Query: "What is the minimum attendance required to write exams?"

Observation: The sentence containing 65% attendance scored 0.796, while the sentence containing 75% scored 0.772.

Conclusion: Both sentences were semantically relevant, but the model ranked 65% higher. Similarity scores do not guarantee numerical correctness.

## 3. Negation

Sentences:
- "Students are allowed to use calculators during examinations."
- "Students are not allowed to use calculators during examinations."

Observation: The cosine similarity was 0.8399, even though the two sentences express opposite rules.

Conclusion: Sentences with similar vocabulary can have high similarity despite an important difference in meaning. High similarity does not guarantee that two statements agree.

## 4. Acronyms

Query: "What does CGPA mean?"

Observation: The model retrieved "CGPA is calculated using semester grades and credits" with a score of 0.755. The sentence "Cumulative grade point average represents overall academic performance" ranked second with a score of 0.294.

Conclusion: The model retrieved relevant information, but the full-form explanation ranked below the sentence containing the exact acronym. This illustrates that retrieval ranking may not always prioritize the most useful explanation.

## 5. General Observation: Semantic Search

Semantic search performed well on paraphrased questions and retrieved relevant information for attendance, mobile phone restrictions, and CGPA. However, exact identifiers, numerical distinctions, negation, and ranking the most useful explanation can be challenging.

These experiments motivate hybrid retrieval, which combines semantic search with keyword-based methods such as BM25 to improve retrieval across different query types.



================================================================================
## QUESTION: What is the minimum attendance required for exams?

BEFORE RERANKING: [1, 2, 0, 7, 3]
AFTER RERANKING:  [1, 2, 0, 7, 3]

ANSWER: The minimum attendance required to be eligible for semester examinations is **75%**【college_rules.txt, chunk 1】.
CITATIONS: ['college_rules.txt, chunk 1']
================================================================================
## QUESTION: When must examination applications be submitted?

BEFORE RERANKING: [3, 6, 1, 7, 0]
AFTER RERANKING:  [3, 6, 1, 7, 4]

ANSWER: Examination applications must be submitted before the announced deadline. 【college_rules.txt, chunk 3】
CITATIONS: ['college_rules.txt, chunk 3']
================================================================================
## QUESTION: Can students use mobile phones during exams?

BEFORE RERANKING: [4, 7, 1, 3, 0]
AFTER RERANKING:  [4, 7, 3, 1, 0]

ANSWER: No. Students are not allowed to use mobile phones during examinations. 【college_rules.txt, chunk 4】
CITATIONS: ['college_rules.txt, chunk 4']
================================================================================
## QUESTION: What does CGPA represent?
BEFORE RERANKING: [5, 0, 1, 7, 2]
AFTER RERANKING:  [5, 0, 1, 7, 2]

ANSWER: The CGPA (cumulative grade point average) represents a student's overall academic performance across completed semesters. 【college_rules.txt, chunk 5】
CITATIONS: ['college_rules.txt, chunk 5']
================================================================================
## QUESTION: When should examination fees be paid?

BEFORE RERANKING: [6, 7, 0, 4, 1]
AFTER RERANKING:  [6, 3, 7, 1, 0]

ANSWER: Examination fees must be paid before the deadline announced by the college administration. 【college_rules.txt, chunk 6】
CITATIONS: ['college_rules.txt, chunk 6']

================================================================================
Metric           Before      After
----------------------------------
Hit@5             1.000      1.000
MRR               1.000      1.000

================================================================================
## REFUSAL TEST
## Question: What time does the college library close?
Answer: I cannot answer this from the provided documents.