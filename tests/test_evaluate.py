"""Tests for evaluate.py's code checks (no API calls)."""

import agent
import evaluate

CHUNK_36 = {"source": "LAU", "location": "Artículo 35, p. 25", "articles": "Artículo 35 | Artículo 36"}
CHUNK_PREAMBLE = {"source": "LeyVivienda", "location": "p. 5", "articles": ""}


def test_a_chunk_matches_every_article_it_contains():
    assert evaluate.matches(CHUNK_36, "LAU", "Artículo 36")
    assert not evaluate.matches(CHUNK_36, "LAU", "Artículo 3")  # not a prefix match
    assert not evaluate.matches(CHUNK_36, "LPH", "Artículo 36")


def test_a_chunk_without_articles_matches_by_page():
    assert evaluate.matches(CHUNK_PREAMBLE, "LeyVivienda", "p. 5")


def test_expected_passages_split_on_semicolons():
    row = {"expected_source": "LAU;LeyVivienda", "expected_location": "Artículo 36;Artículo 31"}
    assert evaluate.expected_passages(row) == [("LAU", "Artículo 36"), ("LeyVivienda", "Artículo 31")]


def test_one_source_can_serve_several_locations():
    row = {"expected_source": "LAU", "expected_location": "Artículo 9;Artículo 10"}
    assert evaluate.expected_passages(row) == [("LAU", "Artículo 9"), ("LAU", "Artículo 10")]


RETRIEVED = ("Result 1 [LAU, Artículo 35, p. 25]\nArtículo 35. Resolución... Artículo 36. Fianza. 1. A la celebración...\n\n"
             "Result 2 [LeyVivienda, Artículo 31, p. 36]\nArtículo 31. Información mínima...")
ANSWERABLE = {"id": "q07", "kind": "single", "expected_source": "LAU", "expected_location": "Artículo 36"}
UNANSWERABLE = {"id": "q31", "kind": "unanswerable", "expected_source": "", "expected_location": ""}


def record(answer: str, guardrail=None) -> dict:
    return {"answer": answer, "guardrail": guardrail, "retrieved": RETRIEVED,
            "tool_calls": [{"name": "search", "args": {"query": "fianza"}}]}


def test_a_passage_counts_as_retrieved_inside_a_result_from_the_same_law():
    assert evaluate.passage_retrieved("LAU", "Artículo 36", RETRIEVED)        # inside a chunk's text
    assert evaluate.passage_retrieved("LeyVivienda", "Artículo 31", RETRIEVED)  # in a header
    assert not evaluate.passage_retrieved("LAU", "Artículo 31", RETRIEVED)    # other law's article 31
    assert not evaluate.passage_retrieved("LAU", "Artículo 3", RETRIEVED)


def test_code_checks_spot_refusals_partial_answers_and_the_guardrail():
    refused = evaluate.code_checks(ANSWERABLE, record(agent.REFUSAL, guardrail="answer has no citations"))
    assert refused["refused"] and refused["guardrail_fired"] and not refused["partial"]
    partial = evaluate.code_checks(UNANSWERABLE, record(f"Algo [LAU, Artículo 35, p. 25].\n{agent.PARTIAL_PREFIX} el resto."))
    assert partial["partial"] and not partial["refused"] and partial["expected_retrieved"] is None
    assert evaluate.code_checks(ANSWERABLE, record("x"))["expected_retrieved"] is True


def test_a_refusal_is_graded_in_code_without_the_judge():
    assert evaluate.refusal_verdict(UNANSWERABLE) == {"correct": 1, "supported": 1, "reason": "refusal (code check)"}
    assert evaluate.refusal_verdict(ANSWERABLE)["correct"] == 0


def test_outcomes_for_answerable_questions():
    checks = {"refused": False, "partial": False}
    assert evaluate.outcome(ANSWERABLE, checks, {"correct": 1, "supported": 1}) == "correct"
    assert evaluate.outcome(ANSWERABLE, checks, {"correct": 1, "supported": 0}) == "unsupported"
    assert evaluate.outcome(ANSWERABLE, checks, {"correct": 0, "supported": 1}) == "incorrect"
    assert evaluate.outcome(ANSWERABLE, {"refused": True, "partial": False}, {"correct": 0, "supported": 1}) == "wrong refusal"


def test_outcomes_for_unanswerable_questions_accept_a_marked_partial_answer():
    assert evaluate.outcome(UNANSWERABLE, {"refused": True, "partial": False}, {}) == "correct refusal"
    marked = {"refused": False, "partial": True}
    assert evaluate.outcome(UNANSWERABLE, marked, {"correct": 1, "supported": 1}) == "correct partial"
    assert evaluate.outcome(UNANSWERABLE, marked, {"correct": 1, "supported": 0}) == "answered unanswerable"
    assert evaluate.outcome(UNANSWERABLE, {"refused": False, "partial": False}, {"correct": 0, "supported": 1}) == "answered unanswerable"


def test_suggested_failure_causes():
    hit, miss = {"expected_retrieved": True}, {"expected_retrieved": False}
    assert evaluate.suggested_cause("correct", hit) is None
    assert evaluate.suggested_cause("wrong refusal", miss) == "retrieval miss"
    assert evaluate.suggested_cause("wrong refusal", hit) == "wrong refusal"
    assert evaluate.suggested_cause("incorrect", miss) == "retrieval miss"
    assert evaluate.suggested_cause("incorrect", hit) == "wrong answer from right passage"
    assert evaluate.suggested_cause("unsupported", hit) == "unsupported claim"
    assert evaluate.suggested_cause("answered unanswerable", {"expected_retrieved": None}) == "unsupported claim"


def test_agreement_compares_judge_and_human_on_the_questions_both_graded():
    judge = {"q01": {"correct": 1, "supported": 1}, "q02": {"correct": 0, "supported": 1}, "q03": {"correct": 1, "supported": 0}}
    human = {"q01": {"correct": "1", "supported": "1"}, "q02": {"correct": "1", "supported": "1"}}
    assert evaluate.agreement(judge, human) == {"n": 2, "correct": 0.5, "supported": 1.0}


def test_hit_at_k_needs_an_expected_passage_within_the_first_k_chunks():
    other = {"source": "LPH", "location": "Artículo noveno, p. 9", "articles": "Artículo noveno"}
    hits = [other, CHUNK_36]
    assert not evaluate.hit_at_k(hits, [("LAU", "Artículo 36")], k=1)
    assert evaluate.hit_at_k(hits, [("LAU", "Artículo 36")], k=3)


def test_an_answer_is_reused_only_for_the_same_question_and_configuration():
    q = {"id": "q01", "question": "¿Cuánto dura?"}
    same = {"question": "¿Cuánto dura?", "config": evaluate.config_signature()}
    assert not evaluate.needs_answer(q, same)
    assert evaluate.needs_answer(q, None)
    assert evaluate.needs_answer(q, {**same, "question": "¿Otra?"})
    assert evaluate.needs_answer(q, {**same, "config": "article|otro-modelo|00000000"})
