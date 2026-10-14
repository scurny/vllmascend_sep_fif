import math
import unittest

import torch

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.ema_predictor import (
EMATopkPredictor,
)


class TestEMATopkPredictor(unittest.TestCase):
    """Pure-CPU unit tests for EMATopkPredictor."""
    def test_initial_threshold_is_zero(self):
        predictor = EMATopkPredictor(topk=128, alpha=0.9)
        self.assertEqual(predictor.get_threshold(), 0.0)
    
    def test_first_update_uses_actual_kth(self):
        """First update should set threshold to the min of actual kth scores."""
        predictor = EMATopkPredictor(topk=4, alpha=0.9)
        # [num_queries=2, topk=4]: each query has topk scores
        scores = torch.tensor([[0.5, 0.4, 0.3, 0.2], [0.6, 0.5, 0.4, 0.3]])
        # global min of kth (column 0, the max score per query) = 0.5... wait
        # min(dim=-1) gives min per row = [0.2, 0.3]. Then .min() = 0.2
        # Actually, kth dimension is topk, so min(dim=-1).values = [0.2, 0.3], then min = 0.2
        # But this is the MIN score in topk, not the kth. Let me reconsider.
        # min(dim=-1).values gives the smallest topk score per query.
        # Then .min() gives the global smallest.
        th = predictor.update(scores)
        expected = float(scores.min(dim=-1).values.min())
        self.assertAlmostEqual(th, expected)
    
    def test_ema_smoothing(self):
        """EMA should smooth consecutive threshold updates."""
        predictor = EMATopkPredictor(topk=2, alpha=0.8)
        step1 = torch.tensor([[1.0, 0.9]])
        step2 = torch.tensor([[0.5, 0.4]])
        
        th1 = predictor.update(step1)
        self.assertAlmostEqual(th1, 0.9)  # first update uses actual
        
        th2 = predictor.update(step2)
        expected = 0.8 * 0.9 + 0.2 * 0.4  # alpha * prev + (1-alpha) * actual
        self.assertAlmostEqual(th2, expected)
    
    def test_ema_converges_to_steady_state(self):
        """Given constant scores, EMA should converge to steady value."""
        predictor = EMATopkPredictor(topk=2, alpha=0.5)
        scores = torch.tensor([[1.0, 0.8]])
        thresholds = []
        for _ in range(10):
            th = predictor.update(scores)
            thresholds.append(th)
        
        # EMA with alpha=0.5: converges to 0.8 (the actual kth)
        self.assertAlmostEqual(thresholds[-1], 0.8, places=4)
        
        # Check monotonic convergence (all updates are 0.8, so no oscillation)
        for i in range(1, len(thresholds)):
            self.assertGreaterEqual(thresholds[i], thresholds[i - 1])
    
    def test_reset_clears_threshold(self):
        predictor = EMATopkPredictor(topk=4, alpha=0.9)
        predictor.update(torch.tensor([[0.5, 0.4, 0.3, 0.2]]))
        self.assertGreater(predictor.get_threshold(), 0.0)
        predictor.reset()
        self.assertEqual(predictor.get_threshold(), 0.0)
    
    def test_update_returns_threshold(self):
        """update() should return the new threshold (not None)."""
        predictor = EMATopkPredictor(topk=2, alpha=0.9)
        th = predictor.update(torch.tensor([[0.5, 0.4], [0.3, 0.2]]))
        self.assertIsInstance(th, float)
        self.assertGreater(th, 0.0)
    
    def test_multi_query_collapse(self):
        """Multiple queries should collapse to the lowest kth score."""
        predictor = EMATopkPredictor(topk=3, alpha=0.9)
        scores = torch.tensor([
        [10.0, 9.0, 8.0],  # query A: high scores
        [1.0, 0.9, 0.8],  # query B: low scores
        ])
        th = predictor.update(scores)
        # global min across all queries' smallest topk score = 0.8
        expected = float(scores.min(dim=-1).values.min())
        self.assertAlmostEqual(th, expected)
    
    def test_invalid_alpha_rejected(self):
        with self.assertRaises(AssertionError):
            EMATopkPredictor(topk=4, alpha=0.0)
        with self.assertRaises(AssertionError):
            EMATopkPredictor(topk=4, alpha=1.0)
        with self.assertRaises(AssertionError):
            EMATopkPredictor(topk=4, alpha=-0.1)
    
    def test_different_topk_sizes(self):
        for topk in [1, 4, 64, 256, 2048]:
            predictor = EMATopkPredictor(topk=topk, alpha=0.9)
            scores = torch.rand(2, topk)
            th = predictor.update(scores)
            self.assertIsInstance(th, float)
            self.assertTrue(0 <= th <= 1.0)
    
    def test_get_threshold_after_reset_and_reupdate(self):
        """After reset, should behave like new."""
        predictor = EMATopkPredictor(topk=2, alpha=0.9)
        scores = torch.tensor([[0.5, 0.4]])
        
        th1 = predictor.update(scores)
        predictor.reset()
        th2 = predictor.update(scores)  # fresh start
        
        self.assertAlmostEqual(th1, 0.4)
        self.assertAlmostEqual(th2, 0.4)
    
    def test_large_topk_performance(self):
        """Sanity: large topk should not crash and produce valid threshold."""
        predictor = EMATopkPredictor(topk=2048, alpha=0.9)
        scores = torch.rand(4, 2048)
        th = predictor.update(scores)
        self.assertIsInstance(th, float)
        self.assertTrue(0.0 < th < 1.0)
    
    def test_consecutive_updates_no_nan(self):
        predictor = EMATopkPredictor(topk=8, alpha=0.95)
        for _ in range(20):
            scores = torch.rand(3, 8)
            th = predictor.update(scores)
            self.assertFalse(math.isnan(th))
            self.assertFalse(math.isinf(th))
    
    def test_ema_with_sudden_score_drop(self):
        """EMA should not collapse to 0 on a single zero-score step."""
        predictor = EMATopkPredictor(topk=2, alpha=0.9)
        predictor.update(torch.tensor([[1.0, 0.9]]))
        
        # sudden drop
        th = predictor.update(torch.tensor([[0.01, 0.0]]))
        
        # EMA: 0.9 * 0.9 + 0.1 * 0.0 = 0.81
        self.assertAlmostEqual(th, 0.9 * 0.9 + 0.1 * 0.0)

if __name__ == "__main__":
    unittest.main()