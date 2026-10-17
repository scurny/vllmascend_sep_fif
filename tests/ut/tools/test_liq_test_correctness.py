import pytest
import torch

from tools.liq_test_correctness import (
    assert_prefetch_payload,
    assert_sequence_padded_scores,
    assert_topk_scores,
    assert_topk_set,
    make_correctness_inputs,
    make_expected_score_lookup,
)


def test_known_topk_and_causal_decoys():
    inputs, expected, hits, positions = make_correctness_inputs("cpu")
    score_lookup = make_expected_score_lookup(inputs)
    assert inputs["actual_seq_lengths_query"].tolist() == [4, 8]
    assert positions[1, 0].item() == 4096
    assert score_lookup[0, 0] == 1 / 16
    assert score_lookup[0, 1536] == 1 / 8
    assert score_lookup[0, -1] == 0
    keys = inputs["key"].reshape(2, 4096, 128).float()
    for token in range(8):
        req, row = divmod(token, 4)
        query = inputs["query"][token].float()
        assert torch.equal(query, query[:1].expand_as(query))
        scores = (keys[req] @ query[0]).relu() * (64 / 1024)
        valid_length = 4093 + row
        if row < 3:
            assert scores[valid_length] > scores[expected[token, 0]].min()
        scores[valid_length:] = -torch.inf
        actual = scores.topk(2048).indices.reshape(1, 1, -1).int()
        assert_topk_set(actual, expected[token:token + 1])
        wanted_hot = torch.arange(1536 + row * 4, 1540 + row * 4)
        assert torch.equal((scores > 3 / 32).nonzero().flatten(), wanted_hot)
        assert hits[req, wanted_hot].all()


def test_topk_accepts_permutations_but_rejects_wrong_members():
    expected = torch.tensor([[[2, 3, 7]]])
    assert_topk_set(torch.tensor([[[7, 2, 3]]], dtype=torch.int32), expected)
    with pytest.raises(AssertionError):
        assert_topk_set(torch.tensor([[[7, 2, 2]]], dtype=torch.int32), expected)


def test_topk_scores_match_the_returned_indices_and_mark_padding():
    score_lookup = torch.tensor([[1.0, 2.0, 3.0]])
    indices = torch.tensor([[[2, 0, -1]]], dtype=torch.int32)
    scores = torch.tensor([[[3.0, 1.0, -torch.inf]]], dtype=torch.float32)

    assert_topk_scores(indices, scores, score_lookup)


@pytest.mark.parametrize("sequence_lengths", ([65, 65], [0, 0], [0, 65]))
def test_sequence_padded_scores_accept_short_and_invalid_rows(sequence_lengths):
    score_lookup = torch.arange(130, dtype=torch.float32).reshape(2, 65)
    indices = torch.full((2, 1, 4), -1, dtype=torch.int32)
    scores = torch.full((2, 1, 4), -torch.inf, dtype=torch.float32)
    for row, sequence_length in enumerate(sequence_lengths):
        valid_count = min(sequence_length, indices.shape[-1])
        indices[row, 0, :valid_count] = torch.arange(valid_count, dtype=torch.int32)
        scores[row, 0, :valid_count] = score_lookup[row, :valid_count]

    assert_sequence_padded_scores(indices, scores, score_lookup, sequence_lengths)


@pytest.mark.parametrize(
    ("indices", "scores"),
    [
        (torch.tensor([[[2, 0]]], dtype=torch.int32), torch.tensor([[[3.0, 2.0]]])),
        (torch.tensor([[[0, 2]]], dtype=torch.int32), torch.tensor([[[3.0, 1.0]]])),
        (torch.tensor([[[2, 0]]], dtype=torch.int32), torch.tensor([[[3.0, float("nan")]]])),
        (torch.tensor([[[2, 0]]], dtype=torch.int32), torch.tensor([[3.0, 1.0]])),
    ],
)
def test_topk_scores_reject_misaligned_values(indices, scores):
    score_lookup = torch.tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(AssertionError):
        assert_topk_scores(indices, scores, score_lookup)


def payload_fixture():
    k = torch.arange(12, dtype=torch.uint8).reshape(2, 3, 2)
    v = (k[:, :, :1] + 100).contiguous()
    hits = torch.tensor([[False, True, False], [False, True, False]])
    slots = torch.tensor([[-1, 1, -1], [-1, 0, -1]], dtype=torch.int32)
    staging = torch.cat((k[:, 1], v[:, 1]), dim=-1).flip(0)
    return staging, slots, k, v, hits


def test_payload_checks_request_identity_and_allows_capacity_misses():
    staging, slots, k, v, hits = payload_fixture()
    assert_prefetch_payload(staging, slots, k, v, hits)
    slots[0, 1] = -1
    assert_prefetch_payload(staging, slots, k, v, hits, require_all=False)
    with pytest.raises(AssertionError):
        assert_prefetch_payload(staging, slots, k, v, hits)


@pytest.mark.parametrize("corruption", ["request", "duplicate", "bounds", "negative", "bytes", "unexpected"])
def test_payload_rejects_corruption(corruption):
    staging, slots, k, v, hits = payload_fixture()
    if corruption == "request":
        slots = slots.flip(0)
    elif corruption == "duplicate":
        slots[0, 1] = slots[1, 1]
    elif corruption == "bounds":
        slots[0, 1] = len(staging)
    elif corruption == "negative":
        slots[0, 1] = -2
    elif corruption == "bytes":
        staging[0, -1] ^= 1
    elif corruption == "unexpected":
        slots[0, 0] = 0
    with pytest.raises(AssertionError):
        assert_prefetch_payload(staging, slots, k, v, hits, require_all=False)
