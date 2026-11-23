import io
import unittest

import torch

from aviy.math import count_parameters, set_seed
from aviy.models.architectures import ARCHITECTURES, ArchitectureConfig, build_model


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        set_seed(7)
        self.config = ArchitectureConfig(
            vocab_size=64, d_model=16, n_layers=1, n_heads=4,
            d_ff=32, context_length=12, num_experts=3,
        )
        self.tokens = torch.randint(0, 64, (2, 8))

    def test_all_registered_architectures(self):
        expected = {"transformer", "rnn", "gru", "lstm", "ssm",
                    "rwkv", "cnn", "moe", "gnn", "hybrid"}
        self.assertEqual(set(ARCHITECTURES.names), expected)

    def test_shapes_gradients_parameters_and_serialization(self):
        for name in ARCHITECTURES.names:
            with self.subTest(architecture=name):
                model = build_model(name, self.config)
                logits = model(self.tokens)
                self.assertEqual(tuple(logits.shape), (2, 8, 64))
                self.assertGreater(count_parameters(model), 0)
                loss = logits.float().square().mean()
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = build_model(name, self.config)
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                restored.eval()
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(self.tokens), restored(self.tokens))

    def test_architectures_are_causal(self):
        tokens = self.tokens.clone()
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 17) % self.config.vocab_size

        for name in ARCHITECTURES.names:
            with self.subTest(architecture=name):
                model = build_model(name, self.config).eval()
                with torch.no_grad():
                    original_logits = model(tokens)
                    changed_logits = model(changed)
                torch.testing.assert_close(
                    original_logits[:, :5],
                    changed_logits[:, :5],
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_parameter_count_matches_parameter_tensors(self):
        for name in ARCHITECTURES.names:
            with self.subTest(architecture=name):
                model = build_model(name, self.config)
                expected = sum(parameter.numel() for parameter in model.parameters())
                self.assertEqual(count_parameters(model), expected)

    def test_rope_configuration_requires_even_head_dimension(self):
        with self.assertRaises(ValueError):
            ArchitectureConfig(vocab_size=32, d_model=15, n_heads=3)

    def test_context_limit(self):
        model = build_model("transformer", self.config)
        with self.assertRaises(ValueError):
            model(torch.zeros((1, 13), dtype=torch.long))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            ArchitectureConfig(d_model=15, n_heads=4)
