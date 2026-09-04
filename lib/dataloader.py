"""
Sim3PluckerData — static .pkl files (pre-generated offline)

Expected .pkl layout:
    <data_dir>/<dataset>_train/
        matches.pkl     list of (2, n_inliers)  int32 arrays
        plucker1.pkl    list of (n_lines, 6)    float32 arrays
        plucker2.pkl    list of (n_lines, 6)    float32 arrays
        R_gt.pkl        list of (3, 3)           float32 arrays
        t_gt.pkl        list of (3, 1)           float32 arrays
        s_gt.pkl        list of float32 scalars
"""
import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


# MAX_LINES: cap per-cloud line count in training to bound the O(N^2) attention
# + dense (n1,n2) match-matrix memory. The FULL dataset has clouds up to ~6000
# lines -> ~31 GB peak, which OOM/cuDNN-faults the trainer on a 32 GB GPU. Keep
# ALL matched inliers, random-subsample the outliers, reindex matches. MAX_LINES=0
# disables (byte-identical). Default 4000 covers the test regime (KITTI ref clouds
# <=~2400 except seq05 5599; solver caps query at 400).
_MAX_LINES = int(os.environ.get('MAX_LINES', '4000'))


def _cap_cloud(pl, matches_ind, row, max_n):
    """Subsample a cloud to <=max_n lines, keeping all matched inliers on `row`
    (0=query/plucker1, 1=ref/plucker2) and reindexing matches."""
    n = pl.shape[0]
    if n <= max_n:
        return pl, matches_ind
    if matches_ind.shape[1] > 0:
        keep = np.unique(matches_ind[row])
        keep = keep[(keep >= 0) & (keep < n)]
    else:
        keep = np.empty(0, dtype=np.int64)
    n_others = max_n - len(keep)
    if n_others > 0:
        mask = np.ones(n, dtype=bool)
        mask[keep] = False
        others = np.nonzero(mask)[0]
        if len(others) > n_others:
            others = np.random.choice(others, n_others, replace=False)
        sel = np.concatenate([keep, others])
    else:
        sel = keep[:max_n]           # more inliers than max_n (rare): truncate
    sel.sort()
    remap = -np.ones(n, dtype=np.int64)
    remap[sel] = np.arange(len(sel))
    pl = pl[sel]
    if matches_ind.shape[1] > 0:
        mi = matches_ind.copy()
        mi[row] = remap[mi[row]]
        mi = mi[:, mi[row] >= 0]     # drop matches whose inlier was truncated
        matches_ind = mi
    return pl, matches_ind

def variable_collate(batch):
    """
    Collate variable-N Plücker samples by zero-padding to the max N in the batch.

    Each item is (matches (N1,N2), plucker1 (N1,C), plucker2 (N2,C), R, t, s).
    Returns stacked tensors with shapes (B,maxN1,maxN2), (B,maxN1,C), (B,maxN2,C),
    (B,3,3), (B,3,1), (B,).

    Padded entries are zero.  The loss ignores padded rows/cols because the
    padded region of the matches matrix is also zero (no GT correspondences).
    """
    matches_l, p1_l, p2_l, R_l, t_l, s_l = zip(*batch)

    B   = len(p1_l)
    C   = p1_l[0].shape[1]
    mn1 = max(p.shape[0] for p in p1_l)
    mn2 = max(p.shape[0] for p in p2_l)

    p1_pad = torch.zeros(B, mn1, C)
    p2_pad = torch.zeros(B, mn2, C)
    m_pad  = torch.zeros(B, mn1, mn2)

    for i, (m, p1, p2) in enumerate(zip(matches_l, p1_l, p2_l)):
        n1, n2 = p1.shape[0], p2.shape[0]
        p1_pad[i, :n1] = torch.as_tensor(p1)
        p2_pad[i, :n2] = torch.as_tensor(p2)
        m_pad[i, :n1, :n2] = torch.as_tensor(m)

    return (
        m_pad,
        p1_pad,
        p2_pad,
        torch.stack([torch.as_tensor(r) for r in R_l]),
        torch.stack([torch.as_tensor(t) for t in t_l]),
        torch.tensor(s_l, dtype=torch.float32),
    )

def load_sim3_data(config, split):
    var_names = ['matches', 'plucker1', 'plucker2', 'R_gt', 't_gt', 's_gt']
    folder = os.path.join(config.data_dir, f'{config.dataset}_{split}')
    data = {}
    for var in var_names:
        with open(os.path.join(folder, f'{var}.pkl'), 'rb') as f:
            data[var] = pickle.load(f, encoding='latin1')
    print(f'[Sim3] loaded {split}: {len(data["t_gt"])} scenes from {folder}')
    return data

class Sim3PluckerData(Dataset):
    """Dataset for Sim(3) Plücker line matching from pre-generated .pkl files.

    Returns per sample:
        matches  (N1, N2) float32 — binary correspondence matrix
        plucker1 (N1, C)  float32
        plucker2 (N2, C)  float32
        R_gt     (3, 3)   float32
        t_gt     (3, 1)   float32
        s_gt     ()       float32 scalar

    N1 and N2 are variable across samples.  Use variable_collate or batch_size=1.
    """

    def __init__(self, phase, config):
        super().__init__()
        self.data       = load_sim3_data(config, phase)
        self.len        = len(self.data['t_gt'])
        self.in_channel = getattr(config, 'in_channel', None)

    def __getitem__(self, index):
        matches_ind = self.data['matches'][index]     # (2, n_inliers)
        plucker1    = self.data['plucker1'][index].copy()    # (n_lines, 6)
        plucker2    = self.data['plucker2'][index].copy()
        R_gt        = self.data['R_gt'][index]
        t_gt        = self.data['t_gt'][index]
        s_gt        = np.float32(self.data['s_gt'][index])

        if _MAX_LINES > 0:
            plucker1, matches_ind = _cap_cloud(plucker1, matches_ind, 0, _MAX_LINES)
            plucker2, matches_ind = _cap_cloud(plucker2, matches_ind, 1, _MAX_LINES)

        if self.in_channel is not None:
            plucker1 = plucker1[:, :self.in_channel]
            plucker2 = plucker2[:, :self.in_channel]

        # Moment normalisation: divide BOTH clouds by plucker2 (query) std.
        # This preserves the scale ratio p1_inlier/p2 ≈ s_gt, which is the
        # key matching signal, while keeping moments in a manageable range.
        # Using p2 std (not p1) avoids contamination by background lines in
        # the KITTI reference cloud which would otherwise compress inlier
        # moments to near-zero, destroying the correspondence signal.
        std2 = float(plucker2[:, :3].std()) + 1e-6
        plucker1[:, :3] /= std2
        plucker2[:, :3] /= std2

        n1, n2 = plucker1.shape[0], plucker2.shape[0]
        matches = np.zeros([n1, n2], dtype=np.float32)
        if matches_ind.shape[1] > 0:
            matches[matches_ind[0, :], matches_ind[1, :]] = 1.0

        return (
            matches.astype('float32'),
            plucker1.astype('float32'),
            plucker2.astype('float32'),
            R_gt.astype('float32'),
            t_gt.astype('float32'),
            s_gt,
        )

    def __len__(self):
        return self.len

# ── LIVE (on-the-fly) generation ──────────────────────────────────────────────
# Restored from commit 9736be6 ("code clean up" removed it) and adapted:
#   * import path moved scripts.generate_synthetic -> generate_synthetic
#   * buffered per EPOCH so the existing BucketBatchSampler still works
#     (it needs all shapes up front; a pair's shape is only known after it is
#     generated, so we materialise one epoch at a time)
#   * the NEXT epoch is generated asynchronously during training, so at
#     192 pairs/s (6 workers) vs 62 samples/s consumed it hides entirely.
# WHY: a fixed set is re-seen 25x (300k) to 154x (50k) over a 240-epoch run,
# and v63 measured the cost -- the 50k subset trails the 300k parent by 1.98
# and widening. Live generation removes repetition entirely and frees the
# ~15 GB the dataset otherwise holds in RAM.

def _live_worker_init(seed):
    import numpy as _np, os as _os
    _np.random.seed((seed * 7919 + _os.getpid()) % (2 ** 32))


def _live_one(_ignored):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    global _LIVE_GEN
    try:
        _LIVE_GEN
    except NameError:
        from generate_synthetic import _generate_pair as _g
        _LIVE_GEN = _g
    d = _LIVE_GEN()
    return (d['matches'], d['plucker1'], d['plucker2'],
            d['R_gt'], d['t_gt'], d['s_gt'])


def _pairs_to_dict(rows):
    return dict(matches=[r[0] for r in rows], plucker1=[r[1] for r in rows],
                plucker2=[r[2] for r in rows], R_gt=[r[3] for r in rows],
                t_gt=[r[4] for r in rows], s_gt=[r[5] for r in rows])


class Sim3LiveData(Sim3PluckerData):
    """Sim3PluckerData over a buffer of FRESH pairs, refilled every epoch.

    Reuses the parent __getitem__ verbatim (canon, moment whitening, capping),
    so live and file-backed training differ ONLY in where the pairs come from.
    """

    def __init__(self, config, epoch_size, workers=6, seed=0):
        Dataset.__init__(self)
        import multiprocessing as _mp
        self.in_channel = getattr(config, 'in_channel', None)
        self.epoch_size = int(epoch_size)
        self._pool = _mp.get_context('spawn').Pool(
            workers, initializer=_live_worker_init, initargs=(seed,))
        self.data = self._gen()
        self.len = self.epoch_size
        self._pending = None
        self._first = True      # constructor already made epoch 1

    def _gen(self):
        # imap, not map: map materialises the ENTIRE 32000-pair result in the
        # parent at once (~1.5 GB) on top of the buffer already held, giving a
        # transient peak at every epoch boundary. imap yields incrementally so
        # only one chunk is in flight. Prefetch was also dropped -- it doubled
        # the resident buffers for a 12% throughput gain, and live runs kept
        # dying at epoch boundaries with the whole process group going at once.
        rows = []
        for r in self._pool.imap_unordered(_live_one, range(self.epoch_size),
                                           chunksize=64):
            rows.append(r)
        return _pairs_to_dict(rows)

    def _start_prefetch(self):
        # map_async, NOT a Python thread. A thread that iterates imap and
        # appends 32000 results holds the GIL throughout, so it does NOT
        # overlap training -- measured 14.7 min/epoch vs 8.5 file-backed
        # (generation running effectively serially). map_async collects in
        # multiprocessing's own handler, which releases the GIL during IPC:
        # measured 9.6 min/epoch, i.e. 12% overhead instead of 73%.
        self._pending = self._pool.map_async(_live_one, range(self.epoch_size),
                                             chunksize=64)

    def regenerate(self):
        """Swap in the prefetched epoch; start generating the next.

        Overlap matters: generating 32000 pairs takes ~7 min, so doing it
        synchronously at every epoch boundary would nearly double epoch time
        (measured: the run stalled 7 min at the start of epoch 0). A THREAD is
        used rather than map_async because pool.imap releases the GIL while
        waiting, and imap accumulates incrementally -- map_async materialised a
        second full copy in the parent, which is the transient peak the earlier
        crashes coincided with.
        """
        if self._first:                      # epoch 1 is already in self.data
            self._first = False
            self._start_prefetch()
            return
        if self._pending is not None:
            self.data = _pairs_to_dict(self._pending.get())
            self._pending = None
        self._start_prefetch()

    def shapes(self):
        return [(len(a), len(b)) for a, b in zip(self.data['plucker1'],
                                                 self.data['plucker2'])]
