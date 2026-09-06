# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import json
import unittest
from pathlib import Path

from validate_checkpoint import (
    FULL_ATTENTION_LAYERS,
    expected_shapes,
    validate_config,
    validate_recipe,
)


HERE = Path(__file__).resolve().parent


class Ling3ValidatorTests(unittest.TestCase):
    def test_expected_checkpoint_inventory_is_exact(self):
        shapes = expected_shapes()
        self.assertEqual(len(shapes), 9283)
        self.assertEqual(
            {layer for layer in range(24) if f"model.layers.{layer}.attention.q_a_proj.weight" in shapes},
            FULL_ATTENTION_LAYERS,
        )

    def test_runtime_and_recipe_match_official_contract(self):
        runtime = json.loads((HERE / "config_tiny_w4a32_kai.json").read_text())
        validate_config(runtime, runtime)
        recipe = json.loads((HERE / "quant_cfg_tiny_w4a32_kai.json").read_text())
        matched = validate_recipe(expected_shapes(), recipe)
        self.assertGreater(len(matched), 9000)
        self.assertNotIn("model.word_embeddings.weight", matched)
        self.assertNotIn("model.layers.1.mlp.gate.weight", matched)


if __name__ == "__main__":
    unittest.main()
