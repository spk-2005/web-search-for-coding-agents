"""Tests for judge evaluation scope and score_needles semantics.

Validates that score_needles checks syntax compilation, required token presence,
and grounded citations, but does NOT evaluate executable correctness or runtime behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

# Ensure scripts dir is on sys.path for direct execution
scripts_dir = Path(__file__).resolve().parents[1]
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from coding_search.judge import JudgeScore, score_needles
from coding_search.tasks import CodingTask


class JudgeSemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.url = "https://docs.example.com/api"
        self.task = CodingTask(
            id="audit",
            repo="",
            outfile="answer.py",
            prompt="Implement the client",
            required=["current_api_field"],
            cite_urls=True,
        )

    def test_runtime_failure_passes_judge(self) -> None:
        """Issue #17 reproduction: code that raises at runtime passes if syntax and grounded tokens are valid."""
        source = "\n".join([
            f"# source: {self.url}",
            'value = "current_api_field"',
            'raise RuntimeError("always broken")',
            "",
        ])
        score = score_needles(
            source,
            self.task,
            observed_urls=[self.url],
            retrieved_text="generic documentation",
        )
        self.assertTrue(score.compiled)
        self.assertTrue(score.passed)
        self.assertEqual(score.output, "NEEDLES_OK")
        self.assertEqual(score.notes, [])

        # Confirm the code is indeed broken when executed at runtime
        with self.assertRaises(RuntimeError):
            exec(compile(source, self.task.outfile, "exec"), {})

    def test_syntax_error_fails(self) -> None:
        """Syntax errors fail compilation and fail the judge."""
        source = "\n".join([
            f"# source: {self.url}",
            "def broken_syntax(",
            '    value = "current_api_field"',
        ])
        score = score_needles(
            source,
            self.task,
            observed_urls=[self.url],
            retrieved_text="generic documentation",
        )
        self.assertFalse(score.compiled)
        self.assertFalse(score.passed)
        self.assertTrue(any("SyntaxError" in note for note in score.notes))

    def test_missing_needle_fails(self) -> None:
        """Missing required gold tokens fail the judge."""
        source = "\n".join([
            f"# source: {self.url}",
            'value = "other_field"',
        ])
        score = score_needles(
            source,
            self.task,
            observed_urls=[self.url],
            retrieved_text="generic documentation",
        )
        self.assertTrue(score.compiled)
        self.assertFalse(score.passed)
        self.assertIn("missing:current_api_field", score.notes)

    def test_uncited_needle_fails(self) -> None:
        """Presence of needle without preceding comment source URL fails."""
        source = 'value = "current_api_field"'
        score = score_needles(
            source,
            self.task,
            observed_urls=[self.url],
            retrieved_text="generic documentation",
        )
        self.assertTrue(score.compiled)
        self.assertFalse(score.passed)
        self.assertIn("uncited:current_api_field", score.notes)

    def test_ungrounded_citation_fails(self) -> None:
        """Citing a URL that never appeared in search/fetch results fails grounding."""
        source = "\n".join([
            "# source: https://unrelated.com/docs",
            'value = "current_api_field"',
        ])
        score = score_needles(
            source,
            self.task,
            observed_urls=[self.url],
            retrieved_text="generic documentation",
        )
        self.assertTrue(score.compiled)
        self.assertFalse(score.passed)
        self.assertIn("ungrounded:current_api_field", score.notes)

    def test_needle_grounded_in_retrieved_text_passes(self) -> None:
        """If needle text appeared in search/fetch snippets, it passes even with different URL formatting."""
        source = "\n".join([
            f"# source: {self.url}/different/path",
            'value = "current_api_field"',
        ])
        score = score_needles(
            source,
            self.task,
            observed_urls=[],
            retrieved_text="Contains current_api_field in the snippet",
        )
        self.assertTrue(score.compiled)
        self.assertTrue(score.passed)

    def test_empty_submission_fails(self) -> None:
        """Empty submission fails both compilation and judge."""
        score = score_needles("", self.task)
        self.assertFalse(score.compiled)
        self.assertFalse(score.passed)
        self.assertEqual(score.notes, ["empty submission"])

    def test_score_dict_format_backward_compatibility(self) -> None:
        """Ensure to_dict returns exactly the expected keys and types."""
        score = JudgeScore(passed=True, compiled=True, output="NEEDLES_OK", notes=[])
        d = score.to_dict()
        self.assertEqual(d, {
            "passed": True,
            "compiled": True,
            "output": "NEEDLES_OK",
            "notes": [],
        })


if __name__ == "__main__":
    unittest.main()
