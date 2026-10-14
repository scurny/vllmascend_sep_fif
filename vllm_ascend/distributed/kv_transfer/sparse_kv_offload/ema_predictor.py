import torch


class EMATopkPredictor:
    """EMA 预测 indexer 输出的第 k 大 score，用于触发 prefetch。
    利用 top-k score 在相邻 decode step 之间的数值相关性，
    用指数移动平均 (EMA) 预测下一 step 的阈值，在 indexer 算分阶段
    就能判断哪些 token 大概率入选 topk，不等 topk 结果出来就触发搬运。
    参考 ECHO (OSDI'26) Section 5.1。
    """

    def __init__(self, topk: int, alpha: float = 0.9):
        """初始化 EMA 预测器。
        Args:
            topk: 稀疏 attention 的 topk 数量。
            alpha: EMA 平滑系数 (0~1)，越大越依赖历史。
        """
        assert 0 < alpha < 1, "alpha must be in (0, 1)"
        self._alpha = alpha
        self._topk = topk
        self._threshold: float | None = None

    def update(self, actual_topk_values: torch.Tensor) -> float:
        """用本步实际 token scores 更新 EMA 预测。
        Args:
            actual_topk_values: 当前 decode step 中 indexer 实际输出的
                topk scores，形状为 [num_queries, topk]。
                取所有 query 中第 k 大 score 用于更新。
        Returns:
            更新后的 EMA 阈值。
        """
        # 取所有 query 中最小的第 k 大 score（最保守的估计）
        kth_scores = actual_topk_values.min(dim=-1).values  # [num_queries]
        actual_kth = kth_scores.min().item()  # scalar: 所有 query 中最小的第 k 大
        if self._threshold is None:
            self._threshold = actual_kth
        else:
            self._threshold = self._alpha * self._threshold + (1 - self._alpha) * actual_kth
        return self._threshold

    def get_threshold(self) -> float:
        """获取当前 EMA 预测阈值。
        Returns:
            当前 EMA 阈值。若还未更新过则返回 0.0（不触发 prefetch）。
        """
        return self._threshold if self._threshold is not None else 0.0

    def reset(self) -> None:
        """重置预测器，清空历史阈值。"""
        self._threshold = None
