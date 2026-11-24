import unittest

import torch

from aviy.math import (
    count_parameters,
    ensure_finite,
    get_device,
    set_seed,
    stable_softmax,
    tensor_summary,
    xavier_initialize,
)


class MathFoundationTests(unittest.TestCase):
    def test_softmax_is_finite_and_normalized(self):
        x = torch.tensor([[10000.0, 10001.0, 9999.0]])
        result = stable_softmax(x)
        self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(result.sum(-1), torch.ones(1))

    def test_softmax_matches_pytorch_reference(self):
        for dtype in (torch.float32, torch.float64):
            values = torch.tensor(
                [[-1000.0, -999.0, -998.0], [1000.0, 1001.0, 1002.0]],
                dtype=dtype,
            )
            actual = stable_softmax(values)
            expected = torch.softmax(values, dim=-1)
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)

    def test_float64_softmax_preserves_precision(self):
        values = torch.tensor([[1.0, 1.0 + 1e-10]], dtype=torch.float64)
        actual = stable_softmax(values)
        expected = torch.softmax(values, dim=-1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_softmax_preserves_dtype(self):
        result = stable_softmax(torch.randn(2, 3, dtype=torch.float16))
        self.assertEqual(result.dtype, torch.float16)

    def test_nonfinite_tensor_rejected(self):
        with self.assertRaises(ValueError):
            ensure_finite(torch.tensor([float("nan")]))

    def test_seed_reproduces_torch_randomness(self):
        set_seed(123)
        first = torch.rand(5)
        set_seed(123)
        second = torch.rand(5)
        torch.testing.assert_close(first, second)

    def test_device_auto(self):
        self.assertEqual(
            get_device("auto").type, "cuda" if torch.cuda.is_available() else "cpu"
        )

    def test_device_cpu_and_invalid_device(self):
        self.assertEqual(get_device("cpu").type, "cpu")
        with self.assertRaises(ValueError):
            get_device("quantum")
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                get_device("cuda")
        if torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                get_device(f"cuda:{torch.cuda.device_count()}")

    def test_parameter_count_and_summary(self):
        layer = torch.nn.Linear(3, 2)
        self.assertEqual(count_parameters(layer), 8)
        self.assertEqual(tensor_summary(torch.ones(2, 3))["shape"], [2, 3])

    def test_initialization(self):
        layer = torch.nn.Linear(8, 8)
        xavier_initialize(layer)
        self.assertTrue(torch.isfinite(layer.weight).all())
