import torch

from models.modules.attention import (
    AlternatingAttention,
    WindowedAlternatingAttention,
    build_window_indices,
)


def test_window_indices_are_centred_and_clamped():
    """A centred window of size 3 keeps interior neighbours and clamps at the edges."""
    indices = build_window_indices(
        length=5, window_size=3, dilation=1, include_self=True, shift=0, device=torch.device("cpu")
    )
    assert indices.shape == (5, 3)
    assert indices[2].tolist() == [1, 2, 3]
    assert indices[0].tolist() == [0, 0, 1]
    assert indices[4].tolist() == [3, 4, 4]


def test_window_indices_apply_dilation_and_shift():
    """Dilation spaces the offsets out and shift translates the whole window."""
    dilated = build_window_indices(
        length=9, window_size=3, dilation=2, include_self=True, shift=0, device=torch.device("cpu")
    )
    assert dilated[4].tolist() == [2, 4, 6]

    shifted = build_window_indices(
        length=9, window_size=3, dilation=1, include_self=True, shift=2, device=torch.device("cpu")
    )
    assert shifted[4].tolist() == [5, 6, 7]


def test_window_size_is_clamped_to_axis_length():
    """A window larger than the axis degenerates to the whole axis instead of failing."""
    indices = build_window_indices(
        length=3, window_size=11, dilation=1, include_self=True, shift=0, device=torch.device("cpu")
    )
    assert indices.shape == (3, 3)


def test_full_window_matches_unwindowed_attention_on_unclamped_query():
    """With a window spanning every channel, the centre query matches full attention.

    Only the centre query is compared: at the edges the window is clamped, which
    duplicates keys and legitimately changes the softmax.
    """
    torch.manual_seed(0)
    channels, patches, dim, heads = 7, 3, 16, 4

    windowed = WindowedAlternatingAttention(
        dim=dim, num_heads=heads, num_channels=channels, block_idx=0,
        window_size_spatial=channels, dilation_spatial=1, include_self=True,
        qkv_bias=True, qk_norm=False,
    ).eval()
    unwindowed = AlternatingAttention(
        dim=dim, num_heads=heads, num_channels=channels, block_idx=0, qkv_bias=True, qk_norm=False,
    ).eval()
    unwindowed.load_state_dict(windowed.state_dict(), strict=False)

    x = torch.randn(2, channels * patches, dim)
    with torch.no_grad():
        got = windowed(x).view(2, channels, patches, dim)
        want = unwindowed(x).view(2, channels, patches, dim)

    centre = (channels - 1) // 2
    torch.testing.assert_close(got[:, centre], want[:, centre], atol=1e-5, rtol=1e-5)


def test_alternating_schedule_flips_axis_between_blocks():
    """Even blocks attend across channels and odd blocks across time."""
    even = WindowedAlternatingAttention(dim=8, num_heads=2, num_channels=4, block_idx=0)
    odd = WindowedAlternatingAttention(dim=8, num_heads=2, num_channels=4, block_idx=1)
    assert even.spatial_pass
    assert not odd.spatial_pass


def test_axial_mode_splits_blocks_into_halves():
    """Axial mode runs every spatial block before every temporal block."""
    passes = [
        WindowedAlternatingAttention(
            dim=8, num_heads=2, num_channels=4, block_idx=i, total_blocks=6, use_axial_mode=True
        ).spatial_pass
        for i in range(6)
    ]
    assert passes == [True, True, True, False, False, False]


def test_padded_queries_are_zeroed_and_padded_keys_are_ignored():
    """Padded tokens produce zero output and cannot influence real tokens."""
    torch.manual_seed(0)
    channels, patches, dim = 6, 4, 16
    attention = WindowedAlternatingAttention(
        dim=dim, num_heads=4, num_channels=channels, block_idx=0, window_size_spatial=channels
    ).eval()

    x = torch.randn(1, channels * patches, dim)
    mask = torch.ones(1, channels * patches, dtype=torch.int)
    mask.view(1, channels, patches)[:, 4:] = 0

    with torch.no_grad():
        out = attention(x, mask).view(1, channels, patches, dim)

        perturbed = x.clone().view(1, channels, patches, dim)
        perturbed[:, 4:] = torch.randn_like(perturbed[:, 4:])
        out_perturbed = attention(perturbed.view(1, channels * patches, dim), mask)
        out_perturbed = out_perturbed.view(1, channels, patches, dim)

    assert torch.all(out[:, 4:] == 0)
    torch.testing.assert_close(out[:, :4], out_perturbed[:, :4], atol=1e-6, rtol=1e-6)
