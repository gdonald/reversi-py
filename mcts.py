import math, numpy as np, torch, copy


class Node:
    __slots__ = ("P", "N", "W", "Q", "children", "legal")

    def __init__(self, prior, legal):
        self.P, self.N, self.W, self.Q = prior, 0, 0.0, 0.0
        self.children = {}
        self.legal = legal


def softmax_t(x, T):
    x = x - x.max()
    e = np.exp(x / max(T, 1e-8))
    return e / e.sum()


class MCTS:
    def __init__(self, model, sims=200, cpuct=1.5, device="cpu"):
        self.f = model.eval()
        self.sims = sims
        self.cpuct = cpuct
        self.device = device

    def run(self, env, temp_moves=10, sims=None):
        sims = sims or self.sims
        root, _ = self._expand(env, node=None, add_noise=True)
        for _ in range(sims):
            self._simulate(copy.deepcopy(env), root)
        visits = np.zeros(65, dtype=np.float32)
        for a, ch in root.children.items():
            visits[a] = ch.N
        return visits

    def _simulate(self, env, node):
        if not node.children:
            _, v = self._expand(env, node=node, add_noise=False)
            return -v

        a, ch = self._select(node)
        env.step(a)
        v = self._simulate(env, ch)

        ch.N += 1
        ch.W += v
        ch.Q = ch.W / ch.N
        return -v

    def _select(self, node):
        sN = math.sqrt(max(1, node.N))
        best, score = None, -1e9

        for a, ch in node.children.items():
            if node.legal[a] == 0:
                continue

            u = self.cpuct * node.P[a] * sN / (1 + ch.N)
            val = ch.Q + u

            if val > score:
                best, score = (a, ch), val

        return best

    @torch.no_grad()
    def _expand(self, env, node=None, add_noise=False):
        x = torch.from_numpy(env.obs()).unsqueeze(0).to(self.device)
        logits, v = self.f(x)
        p = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        legal = env.legal_mask()
        p = p * legal
        s = p.sum()
        p = p / s if s > 0 else legal / max(1.0, legal.sum())

        if add_noise:
            idx = np.where(legal > 0)[0]
            noise = np.random.dirichlet([0.3] * len(idx))
            p[idx] = 0.75 * p[idx] + 0.25 * noise

        if node is None:
            node = Node(prior=p, legal=legal)
        else:
            node.P, node.legal = p, legal

        if not node.children:
            for a in np.where(legal > 0)[0]:
                node.children[a] = Node(prior=p[a], legal=None)

        node.N = node.N
        return node, float(v.item())
