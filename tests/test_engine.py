"""Real tests for the TalentMatch engine."""

import math

import pytest

from talentmatch.engine import TfIdf, cosine, extract_skills, gap_analysis, match, rank

# --- tiny fixtures ---------------------------------------------------------

RESUME_BACKEND = """\
Backend developer. Java, Spring Boot, PostgreSQL, Docker, REST API,
microservices, Git, Agile.
"""

JD_BACKEND = """\
We need a backend developer with Java, Spring Boot, PostgreSQL, Docker,
REST API, and microservices experience. Git and Agile required.
"""

RESUME_CHEF = """\
Executive chef with 10 years in French cuisine. Knife skills, menu design,
pastry, wine pairing, kitchen management.
"""


@pytest.fixture
def pair():
    return match(RESUME_BACKEND, JD_BACKEND, resume_name="backend", jd_name="jd")


# --- score behaviour -------------------------------------------------------

def test_identical_documents_score_near_100():
    r = match(RESUME_BACKEND, RESUME_BACKEND, resume_name="a", jd_name="b")
    assert r.score == pytest.approx(100.0, abs=0.01)
    assert r.cosine_similarity == pytest.approx(1.0, abs=1e-9)


def test_unrelated_pair_scores_low():
    r = match(RESUME_CHEF, JD_BACKEND, resume_name="chef", jd_name="jd")
    assert r.score < 15.0
    assert r.matched_skills == []


def test_matching_pair_scores_high(pair):
    assert pair.score >= 70.0
    assert pair.cosine_similarity > 0.4


# --- skill extraction ------------------------------------------------------

def test_extract_skills_multiword_phrases():
    found = extract_skills("Built APIs with Spring Boot and Tailwind CSS on AWS.")
    assert "spring boot" in found
    assert "tailwind css" in found
    assert "aws" in found


def test_extract_skills_no_false_positive_on_component():
    # "spring boot" should not report bare "spring" as a separate skill.
    found = extract_skills("Experienced with Spring Boot only.")
    assert "spring boot" in found
    assert "spring" not in found


def test_missing_skills_correct_on_fixture(pair):
    jd_skills = {"java", "spring boot", "postgresql", "docker", "rest api",
                 "microservices", "git", "agile"}
    resume_skills = extract_skills(RESUME_BACKEND)
    assert resume_skills == jd_skills
    assert pair.missing_skills == []
    assert set(pair.matched_skills) == jd_skills


def test_missing_skills_lists_the_gap():
    resume = "Java developer. Spring Boot, Git."
    jd = "Java developer. Spring Boot, Kubernetes, Docker, Git."
    r = match(resume, jd)
    assert r.missing_skills == ["docker", "kubernetes"]
    assert "kubernetes" in r.missing_skills


# --- coverage --------------------------------------------------------------

def test_coverage_by_category(pair):
    cov = pair.coverage_by_category
    assert cov["languages"]["coverage"] == pytest.approx(1.0)
    assert cov["backend"]["matched"] == cov["backend"]["total"] > 0


# --- ranking ---------------------------------------------------------------

def test_ranking_puts_best_candidate_first():
    resumes = {
        "chef": RESUME_CHEF,
        "backend_dev": RESUME_BACKEND,
        "frontend_dev": "React developer. JavaScript, TypeScript, React, Tailwind CSS, Jest.",
    }
    results = rank(resumes, JD_BACKEND, jd_name="jd")
    assert [r.resume_name for r in results][0] == "backend_dev"
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# --- gap analysis ----------------------------------------------------------

def test_gap_analysis_verdict_and_add_list():
    r = match("Java developer. Spring Boot.", "Java developer. Spring Boot, Docker, Kubernetes.")
    gaps = gap_analysis(r, target=90.0)
    assert gaps["current_score"] == pytest.approx(r.score)
    assert gaps["target"] == 90.0
    assert set(gaps["add_these_skills"]) == {"docker", "kubernetes"}
    assert gaps["gap"] > 0


def test_gap_analysis_already_at_target():
    r = match(RESUME_BACKEND, RESUME_BACKEND)
    gaps = gap_analysis(r, target=90.0)
    assert gaps["gap"] == 0.0
    assert "Already" in gaps["verdict"]


# --- vectoriser maths ------------------------------------------------------

def test_cosine_orthogonal_vectors():
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_identical_vectors():
    assert cosine({"a": 0.5, "b": 0.25}, {"a": 0.5, "b": 0.25}) == pytest.approx(1.0)


def test_tfidf_idf_downweights_common_terms():
    docs = [["python", "the"], ["java", "the"], ["go", "the"]]
    vec = TfIdf().fit(docs)
    assert vec.idf["the"] < vec.idf["python"]
    assert vec.idf["the"] == pytest.approx(1.0)  # in every doc → log(1)+1
