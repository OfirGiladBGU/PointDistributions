import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

Pair = Tuple[int, int]
Cluster = List[Pair]

###############
# PRE-PROCESS #
###############
def image_density_sampling(image_path, m, invert=True, gamma=1.8, seed=0):
    # Build P (|P|=m) by sampling from an image-driven distribution (ρ=1 per discrete point)
    img = Image.open(image_path).convert('L')
    arr = np.asarray(img, dtype=np.float64) / 255.0
    if invert:
        arr = 1.0 - arr              # darker => denser points (stippling)
    arr = np.clip(arr, 0, 1) ** gamma
    probs = arr / (arr.sum() + 1e-12)

    H, W = probs.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(H*W, size=m, p=probs.ravel(), replace=True)
    y = idx // W
    x = idx % W

    # jitter to spread points within pixels; normalize to [0,1]^2
    Px = (x + rng.random(m)) / W
    Py = (y + rng.random(m)) / H
    P = np.stack([Px, Py], axis=1)
    return P


def init_sites(n, seed=1):
    # Build S (no special constraint in the paper for seeding)
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def build_S_P_C_from_image(image_path, n_sites, points_per_site=256, invert=True, gamma=1.8, seed=0):
    # Paper-faithful: discrete CCVT with constant per-site capacity and adaptive distribution via P
    m = n_sites * points_per_site              # ensure m/n is an integer (paper suggests >=256)
    P = image_density_sampling(image_path, m, invert=invert, gamma=gamma, seed=seed)
    S = init_sites(n_sites, seed=seed+1)
    C = np.full(n_sites, points_per_site, dtype=int)  # capacities per site (vector), sum(C)=m
    return S, P, C


###########
# PROCESS #
###########

# Program 6
def Initialization(S, P, C):
    # V ← coherent assignment from P to S, preserving C
    # S' ← S  // the set of sites with capacity not yet filled

    # Vi ← ∅
    n = S.shape[0]
    V = [[] for _ in range(n)]

    # S' ← S // the set of sites with capacity not yet filled
    S_prime = set(range(n))

    # foreach p ∈ P
    for pi in range(P.shape[0]):
        p = P[pi]

        # si ← arg min_{s ∈ S'} |s - p|
        idxs = np.fromiter(S_prime, dtype=int)
        diffs = S[idxs] - p
        d2 = np.einsum('ij,ij->i', diffs, diffs)  # squared distances (argmin identical to |·|)
        si = idxs[np.argmin(d2)]

        # s(p) ← si
        # V_i ← V_i ∪ p
        V[si].append(pi)

        # if |V_i| ≥ c(si) // si has reached its capacity
        #   remove si from S'
        if len(V[si]) >= C[si]:
            S_prime.remove(si)

    # return V
    return V


# Program 5
def SelectSitePairClusters(S, last_swapped_pairs: Optional[List[Pair]] = None) -> List[Cluster]:
    Υb = SelectSitePairClustersCachedStage(S, last_swapped_pairs or [])
    if len(Υb) == 0:
        Υb = SelectSitePairClustersFullStage(S)
    return Υb


def SelectSitePairClustersFullStage(S) -> List[Cluster]:
    n = len(S)
    Υb: List[Cluster] = []
    for k in range(n):
        Υ: Cluster = []
        # for(i ← 0, j ← k - 1; i < j; i ← i + 1, j ← j − 1)  Υ ← Υ ∪ (s_i, s_j)
        i, j = 0, k - 1
        while i < j:
            Υ.append((i, j))
            i += 1
            j -= 1
        # for(i ← k, j ← n - 1; i < j; i ← i + 1, j ← j − 1)  Υ ← Υ ∪ (s_i, s_j)
        i, j = k, n - 1
        while i < j:
            Υ.append((i, j))
            i += 1
            j -= 1
        Υb.append(Υ)
    return Υb  # {Υ}


def SelectSitePairClustersCachedStage(S, Υinit: List[Pair]) -> List[Cluster]:
    Υb: List[Cluster] = []
    clusters_sites: List[set] = []

    for (si, sj) in Υinit:
        added = False
        # foreach Υ ∈ Υb in increasing size
        for idx in sorted(range(len(Υb)), key=lambda t: len(Υb[t])):
            used = clusters_sites[idx]
            # if neither si nor sj belongs to any pair ∈ Υ  // no conflict
            if (si not in used) and (sj not in used):
                Υb[idx].append((si, sj))          # Υ ← Υ ∪ (s_i, s_j)
                used.update([si, sj])
                added = True
                break
        # if not added  // conflict with all existing clusters
        if not added:
            Υb.append([(si, sj)])                  # start a new cluster
            clusters_sites.append({si, sj})
    return Υb

# Program 3
def SiteSwap(si, sj, S, P, C, V):
    # U ← V_i ∪ V_j
    U = np.array(V[si] + V[sj], dtype=int)
    if U.size == 0:
        return False  # nothing to do

    # For each p ∈ U, ΔE(p, si→sj) = ||p - s_j||^2 − ||p - s_i||^2
    PU = P[U]
    d2_i = np.einsum('md,md->m', PU - S[si], PU - S[si])
    d2_j = np.einsum('md,md->m', PU - S[sj], PU - S[sj])
    delta = d2_j - d2_i

    # Select exactly C[sj] smallest ΔE to assign to sj (median-threshold selection)
    k = int(C[sj])
    if k <= 0:
        take_j = np.array([], dtype=int)
    elif k >= U.size:
        take_j = np.arange(U.size, dtype=int)
    else:
        take_j = np.argpartition(delta, k - 1)[:k]

    mask_j = np.zeros(U.size, dtype=bool)
    mask_j[take_j] = True

    # V_j ← selected ; V_i ← U \ selected
    V[sj] = U[mask_j].tolist()
    V[si] = U[~mask_j].tolist()

    # optional boolean return (whether anything changed)
    return True


# Program 2
def FastCCVT(S, P, C, max_workers=None, max_iterations=10):
    # input: sites set S, points set P, and capacity constraints C
    #        where \sigma_(si ∈ S) C(si) = |P|.
    # output: the Voronoi set V

    # Initialization(S,P,C) is Program 6
    V = Initialization(S, P, C)

    stable = False
    last_swapped_pairs = []  # for cached-stage clustering (Program 5)
    iteration = 0
    while not stable and iteration < max_iterations:
        print(f"FastCCVT iteration {iteration}")
        iteration += 1

        stable = True
        changed = False

        # Program 5: use cached stage if we have swaps from last iter
        Yb = SelectSitePairClusters(S, last_swapped_pairs)

        swapped_this_round = []

        for Y in Yb:
            # Run all pairs in this cluster in parallel (disjoint sites ⇒ safe writes to V)
            with ThreadPoolExecutor(max_workers=(max_workers or len(Y) or 1)) as ex:
                futures = {ex.submit(SiteSwap, si, sj, S, P, C, V): (si, sj) for (si, sj) in Y}
                for fut in as_completed(futures):
                    si, sj = futures[fut]
                    did_swap = fut.result()  # expect bool from SiteSwap
                    if did_swap:
                        changed = True
                        swapped_this_round.append((si, sj))

        last_swapped_pairs = swapped_this_round  # feeds cached stage next loop
        if changed:
            stable = False

    return V


################
# POST-PROCESS #
################
def visualize_ccvt(S, P, V, point_size=1, site_size=80, alpha=0.85,
                   figsize=(6,6), dpi=120, title=None, show=True, origin='image'):
    m = P.shape[0]
    labels = np.empty(m, dtype=int)
    for i, idxs in enumerate(V):
        if idxs:
            labels[np.asarray(idxs, dtype=int)] = i

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(P[:,0], P[:,1], c=labels, s=point_size, alpha=alpha, linewidths=0)
    ax.scatter(S[:,0], S[:,1], marker='x', s=site_size, linewidths=1.5, c='k')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)

    # Flip Y so (0,0) is top-left like an image
    if origin == 'image':
        ax.invert_yaxis()

    if show:
        plt.show()
    return fig, ax


# Main
def main():
    S, P, C = build_S_P_C_from_image(image_path, n_sites=128, points_per_site=128, invert=True, gamma=1.6)
    print(f"Input shapes S.shape: {S.shape}, P.shape: {P.shape}, C.shape: {C.shape}")
    V = FastCCVT(S, P, C)
    visualize_ccvt(S, P, V, point_size=1, site_size=80, alpha=0.85, figsize=(6,6), dpi=120, title="Fast CCVT Result", show=True)


if __name__ == "__main__":
    image_path = fr"C:\Users\ofirg\PycharmProjects\PointDistributions\input\Scale.jpg"
    main()
