"""Unit tests for engine v3 OscRNN features."""
from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__file__).rsplit("/engine_v3", 1)[0])

from engine_v3.rate_network import OscRNN, OscRNNConfig, synthetic_wiring
from engine_v3.wiring import assign_dale_signs, delay_from_length, myelin_speed_scale


torch.set_num_threads(1)


def _model(n=48, seed=0, **cfg_kw):
    w = synthetic_wiring(n=n, seed=seed, p_intra=0.1, p_inter=0.04)
    cfg = OscRNNConfig(**cfg_kw)
    torch.manual_seed(seed)
    m = OscRNN(w, n_in=2, out_dim=1, cfg=cfg, readout_mode="left_hemi", input_mode="hemi_split", seed=seed)
    return m, w


def test_ring_buffer_matches_roll():
    m, _ = _model(seed=3)
    m.eval()
    u = torch.randn(32, 8, 2)
    with torch.no_grad():
        y_ring = m.forward_sequence(u)
        y_roll = m.forward_sequence_roll_reference(u)
    err = (y_ring - y_roll).abs().max().item()
    assert err < 1e-5, f"ring vs roll max abs {err}"


def test_stp_off_identical():
    m, _ = _model(seed=5, use_stp=False)
    m.eval()
    u = torch.randn(20, 4, 2)
    with torch.no_grad():
        a = m.forward_sequence(u)
        b = m.forward_sequence_roll_reference(u)
    assert (a - b).abs().max().item() < 1e-5


def test_stp_on_changes_output():
    w = synthetic_wiring(n=40, seed=7, p_inter=0.05)
    torch.manual_seed(7)
    base = OscRNN(w, 2, 1, cfg=OscRNNConfig(use_stp=False), seed=7)
    torch.manual_seed(7)
    stp = OscRNN(w, 2, 1, cfg=OscRNNConfig(use_stp=True), seed=7)
    # copy weights
    stp.load_state_dict(base.state_dict(), strict=False)
    u = torch.rand(40, 4, 2)
    with torch.no_grad():
        y0 = base.forward_sequence(u)
        y1 = stp.forward_sequence(u)
    assert (y0 - y1).abs().max().item() > 1e-4


def test_dale_no_sign_flip_across_neuron():
    w = synthetic_wiring(n=32, seed=11)
    signs = assign_dale_signs(32, p_inhibit=0.25, seed=11)
    m = OscRNN(w, 2, 1, cfg=OscRNNConfig(use_dale=True), neuron_signs=signs, seed=11)
    with torch.no_grad():
        wi = m._intra_weights()
        wj = m._inter_weights()
    # all out-edges of a neuron share the neuron's sign
    for e, src in enumerate(m.intra_src.tolist()):
        assert float(wi[e] * signs[src]) >= -1e-6 or abs(float(wi[e])) < 1e-8 or np.sign(float(wi[e])) == np.sign(signs[src]) or float(wi[e]) == 0.0
        if abs(float(wi[e])) > 1e-8:
            assert np.sign(float(wi[e])) == np.sign(signs[src])
    for e, src in enumerate(m.inter_src.tolist()):
        if abs(float(wj[e])) > 1e-8:
            assert np.sign(float(wj[e])) == np.sign(signs[src])


def test_dual_tau_runs():
    m, _ = _model(seed=13, use_dual_tau=True)
    u = torch.randn(16, 4, 2)
    y = m.forward_sequence(u)
    assert torch.isfinite(y).all()


def test_homeostasis_runs():
    m, _ = _model(seed=15, use_homeostasis=True)
    b0 = m.b.detach().clone()
    u = torch.randn(24, 4, 2)
    _ = m.forward_sequence(u)
    assert not torch.equal(b0, m.b.detach())


def test_sparse_vs_dense_masked():
    """Sparse edge forward ≡ dense masked on same nnz (N small)."""
    n = 24
    w = synthetic_wiring(n=n, seed=21, p_intra=0.15, p_inter=0.08)
    m, _ = _model(n=n, seed=21)  # rebuild with same seed wiring? use w
    torch.manual_seed(21)
    m = OscRNN(w, 2, 1, cfg=OscRNNConfig(), seed=21)
    # build dense W from edges
    Wi = torch.zeros(n, n)
    Wj = torch.zeros(n, n)
    Wi[m.intra_dst, m.intra_src] = m.w_intra.detach()  # note: scatter uses src→dst with x@W.t style
    # Our scatter: msgs = x[:,src]*w; index_add dst → rec[b,dst]+=x[b,src]*w
    # Dense equiv: rec = x @ W.t() with W[dst,src]=w  → (x @ W.t())[b,dst] = sum_src x[b,src]*W[dst,src]
    Wi = torch.zeros(n, n)
    Wi[m.intra_dst, m.intra_src] = m.w_intra.detach()
    Wj_raw = torch.zeros(n, n)
    Wj_raw[m.inter_dst, m.inter_src] = m.w_inter.detach()
    es = torch.zeros(n, n)
    es[m.inter_dst, m.inter_src] = m.inter_sign.detach()

    B, T = 3, 12
    u = torch.randn(T, B, 2)
    m.eval()
    with torch.no_grad():
        # run ring and also manual dense
        x = torch.zeros(B, n)
        hist = torch.zeros(m.dmax + 1, B, n)
        head = -1
        for t in range(T):
            head = (head + 1) % (m.dmax + 1)
            hist[head] = x
            rec_s = m._recurrent(x, hist, head)
            # dense
            rec_d = x @ Wi.t()
            wmag = F.softplus(Wj_raw) * es
            for d in range(1, m.dmax + 1):
                src_t = (head - d) % (m.dmax + 1)
                md = torch.zeros(n, n)
                # mark edges with this delay
                for e in range(m.inter_src.numel()):
                    if int(m.inter_delay[e]) == d:
                        md[m.inter_dst[e], m.inter_src[e]] = 1.0
                rec_d = rec_d + hist[src_t] @ (wmag * md).t()
            assert (rec_s - rec_d).abs().max().item() < 1e-5
            drive = u[t] @ m.Win.t() + m.b
            x = (1 - m.leak) * x + m.leak * torch.tanh(rec_s + drive)


def test_delay_helpers():
    lengths = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
    d = delay_from_length(150.0, lengths, dmax=5, speed_scale=1.0)
    assert 1 <= d <= 5
    assert myelin_speed_scale(fa=0.5) > 0.5


def test_nan_guard():
    m, _ = _model(seed=99, nan_guard=True)
    # huge input shouldn't produce NaN outputs
    u = torch.full((10, 2, 2), 1e6)
    y = m.forward_sequence(u)
    assert torch.isfinite(y).all()


def test_feature_arms_multi_step_train():
    """Each feature flag must survive several Adam steps (no double-backward).

    STP used to keep stp_u/stp_x with grad_fn across batches via in-place fill_,
    which crashed on the 2nd backward. This guards that regression.
    """
    from engine_v3.train import feature_cfg, train_task, DEFAULT_HP

    w = synthetic_wiring(n=32, seed=0, p_intra=0.1, p_inter=0.05)
    hp = dict(
        DEFAULT_HP,
        epochs=2,
        steps_per_epoch=3,
        batch_xor=4,
        seq_xor=16,
        emb_dim=8,
    )
    for feat in ("BASE", "STP", "DUAL_TAU", "DALE", "HOMEO", "ALL"):
        cfg = feature_cfg(feat, leak=0.4)
        r = train_task(
            w, "XOR_HARD", 3300, hp, feature=feat, cfg=cfg, arm="SYNTH", record_curve=False
        )
        assert not r.get("poisoned"), f"{feat} poisoned: {r.get('abort_reason')}"
        assert r.get("metric") is not None and np.isfinite(r["metric"]), feat


def test_stp_state_is_leaf_across_batches():
    """After forward+backward, a new forward must start from leaf STP tensors."""
    m, _ = _model(seed=41, use_stp=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    u = torch.rand(12, 4, 2)
    for _ in range(3):
        opt.zero_grad()
        y = m.forward_sequence(u)
        loss = y.pow(2).mean()
        loss.backward()
        opt.step()
        assert m.stp_u.grad_fn is None, "stp_u must be detached/leaf between batches"
        assert m.stp_x.grad_fn is None, "stp_x must be detached/leaf between batches"
