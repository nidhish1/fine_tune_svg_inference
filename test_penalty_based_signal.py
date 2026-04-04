#!/usr/bin/env python3
import unittest

from penalty_based_signal import (
    ValidationConfig,
    batch_penalty_from_svgs,
    compute_invalidity_penalty,
    penalty_augmented_loss,
    validate_svg,
)


class TestPenaltyBasedSignal(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = ValidationConfig()

    def test_valid_svg_has_zero_penalty(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            '<path d="M10 10 L50 50" fill="#000"/>'
            "</svg>"
        )
        res = validate_svg(svg, self.cfg)
        self.assertTrue(res.is_valid)
        self.assertEqual(compute_invalidity_penalty(res), 0.0)

    def test_invalid_xml_penalty(self) -> None:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><g><path d="M0 0 L10 10"></svg>'
        res = validate_svg(svg, self.cfg)
        self.assertFalse(res.xml_valid)
        self.assertEqual(compute_invalidity_penalty(res), 3.0)

    def test_invalid_root_penalty(self) -> None:
        svg = "<html><body><svg xmlns='http://www.w3.org/2000/svg'/></body></html>"
        res = validate_svg(svg, self.cfg)
        self.assertTrue(res.xml_valid)
        self.assertFalse(res.svg_root_valid)
        self.assertEqual(compute_invalidity_penalty(res), 2.5)  # root + 2 disallowed tags

    def test_disallowed_tag_penalty(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<script>alert(1)</script>"
            '<path d="M0 0 L1 1"/>'
            "</svg>"
        )
        res = validate_svg(svg, self.cfg)
        self.assertEqual(res.disallowed_tag_count, 1)
        self.assertAlmostEqual(compute_invalidity_penalty(res), 0.5, places=6)

    def test_path_overflow_penalty(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            + "".join('<path d="M0 0 L1 1"/>' for _ in range(257))
            + "</svg>"
        )
        res = validate_svg(svg, self.cfg)
        self.assertEqual(res.path_overflow, 1)
        self.assertAlmostEqual(compute_invalidity_penalty(res), 0.05, places=6)

    def test_char_overflow_penalty(self) -> None:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><desc>' + ("a" * 17000) + "</desc></svg>"
        res = validate_svg(svg, self.cfg)
        expected = max(0, len(svg) - self.cfg.max_chars) * 0.001
        self.assertGreater(res.char_overflow, 0)
        self.assertAlmostEqual(compute_invalidity_penalty(res), expected, places=6)

    def test_batch_penalty_mean(self) -> None:
        valid = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 1"/></svg>'
        invalid = '<svg xmlns="http://www.w3.org/2000/svg"><script>x</script></svg>'
        mean_penalty = batch_penalty_from_svgs([valid, invalid], cfg=self.cfg)
        self.assertAlmostEqual(mean_penalty, 0.25, places=6)

    def test_penalty_augmented_loss(self) -> None:
        valid = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0 L1 1"/></svg>'
        invalid = '<svg xmlns="http://www.w3.org/2000/svg"><script>x</script></svg>'
        out = penalty_augmented_loss(
            base_loss=1.0,
            generated_svgs=[valid, invalid],
            lambda_penalty=0.2,
            cfg=self.cfg,
        )
        self.assertAlmostEqual(out["base_loss"], 1.0, places=6)
        self.assertAlmostEqual(out["invalidity_penalty"], 0.25, places=6)
        self.assertAlmostEqual(out["total_loss"], 1.05, places=6)


if __name__ == "__main__":
    unittest.main()
