#!/usr/bin/env python3
import unittest

from svg_semantic_tokenization import (
    lines_to_tokens,
    semantic_tokens_to_svg,
    svg_to_semantic_tokens,
    tokens_to_lines,
)


class TestSvgSemanticTokenization(unittest.TestCase):
    def test_tokenize_contains_expected_fields(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            "<g>"
            '<path fill="#000" stroke-width="2" d="M 10 20 L 30 40 Z"/>'
            "</g>"
            "</svg>"
        )
        tokens = svg_to_semantic_tokens(svg)
        lines = tokens_to_lines(tokens)

        self.assertIn("structure|OPEN_SVG|", lines)
        self.assertIn("structure|OPEN_G|", lines)
        self.assertIn("tag|TAG|PATH", lines)
        self.assertIn("style|FILL|#000", lines)
        self.assertIn("style|STROKE_WIDTH|2", lines)
        self.assertIn("geometry|PATH_D_START|", lines)
        self.assertIn("path_cmd|CMD|M", lines)
        self.assertIn("path_param|X|10", lines)
        self.assertIn("path_param|Y|20", lines)
        self.assertIn("path_cmd|CMD|Z", lines)
        self.assertIn("geometry|PATH_D_END|", lines)
        self.assertIn("structure|CLOSE_SVG|", lines)

    def test_round_trip_reconstruction(self) -> None:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
            '<path fill="#111" d="M 1 2 L 3 4 Z"/>'
            "</svg>"
        )
        tokens = svg_to_semantic_tokens(svg)
        rebuilt = semantic_tokens_to_svg(tokens)
        rebuilt_tokens = svg_to_semantic_tokens(rebuilt)

        self.assertGreater(len(rebuilt), 0)
        self.assertEqual(tokens_to_lines(tokens), tokens_to_lines(rebuilt_tokens))

    def test_lines_conversion(self) -> None:
        svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="10" cy="20" r="5"/></svg>'
        tokens = svg_to_semantic_tokens(svg)
        lines = tokens_to_lines(tokens)
        parsed = lines_to_tokens(lines)
        self.assertEqual(lines, tokens_to_lines(parsed))


if __name__ == "__main__":
    unittest.main()
