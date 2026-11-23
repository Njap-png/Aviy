import unittest

import torch

from aviy.training import (
    build_optimizer,
    causal_language_model_loss,
    clip_gradients,
    l2_regularization,
    warmup_cosine_scheduler,
)
from aviy.training.benchmark import benchmark_optimizers


class OptimizationTests(unittest.TestCase):
    def test_supported_optimizers_update_parameters(self):
        for name in ("sgd", "adam", "adamw"):
            model = torch.nn.Linear(4, 2)
            before = model.weight.detach().clone()
            optimizer = build_optimizer(name, model.parameters(), learning_rate=0.01)
            loss = model(torch.ones(3, 4)).square().mean()
            loss.backward()
            clip_gradients(model, 1.0)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.weight))

    def test_causal_loss_and_regularization(self):
        logits = torch.randn(2, 4, 16, requires_grad=True)
        targets = torch.randint(0, 16, (2, 4))
        loss = causal_language_model_loss(logits, targets, label_smoothing=0.1)
        penalty = l2_regularization([logits], 0.01)
        (loss + penalty).backward()
        self.assertTrue(torch.isfinite(loss))

    def test_scheduler_changes_learning_rate(self):
        model = torch.nn.Linear(2, 2)
        optimizer = build_optimizer("adamw", model.parameters(), learning_rate=0.1)
        scheduler = warmup_cosine_scheduler(optimizer, warmup_steps=2, total_steps=8)
        rates = []
        for _ in range(8):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertGreater(max(rates), min(rates))

    def test_optimizer_benchmark(self):
        result = benchmark_optimizers(seed=12, steps=5, width=8)
        self.assertEqual(set(result), {"sgd", "adam", "adamw"})
        for item in result.values():
            self.assertGreater(item["elapsed_seconds"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(item["final_loss"])))
