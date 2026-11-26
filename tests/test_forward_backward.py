"""Exercise training, optimizer updates, and checkpoint reconstruction."""

import io
import unittest

import torch

from model.bci2a import DualBranchMINet
from utils import load_config


class ForwardBackwardTests(unittest.TestCase):
    def test_training_batch_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        config = load_config()
        model = DualBranchMINet(config).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        x = torch.randn(2, 22, 1000)
        labels = torch.tensor([0, 3])
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        logits = model(x, x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(before[name], value.detach())
            for name, value in model.named_parameters()
        ))
        model.eval()
        with torch.no_grad():
            expected = model(x, x)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = DualBranchMINet(config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        with torch.no_grad():
            actual = restored(x, x)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
