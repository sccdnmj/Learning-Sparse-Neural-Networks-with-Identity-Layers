import torch.nn as nn


def _HSIC(K, L):
    N = K.shape[0]
    ones = torch.ones(N, 1).cuda()
    result = torch.trace(K @ L)
    result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2)))[0][0]
    result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2))[0][0]
    return (1 / (N * (N - 3)) * result)

class CKASR(nn.CrossEntropyLoss):
    def __init__(self, ):
        super(CKASR, self).__init__()

    def _HSIC(self, K, L):
        N = K.shape[0]
        ones = torch.ones(N, 1).double().cuda()
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2)))[0][0]
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2))[0][0]
        return (1 / (N * (N - 3)) * result)

    def forward(self, blocks):
        total = sum([len(blocks[i])*len(blocks[i]) for i in range(len(blocks))])
        hsic_matrix = torch.zeros(total, 3).cuda()
        cnt = 0
        for feats in blocks:
            _num = len(feats)
            for i in range(_num):
                for j in range(_num):
                    feat1, feat2 = feats[i], feats[j]
                    X = feat1.flatten(1).double()
                    K = X @ X.t()
                    K.fill_diagonal_(0.0)

                    Y = feat2.flatten(1).double()
                    L = Y @ Y.t()
                    L.fill_diagonal_(0.0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    hsic_matrix[cnt, 1] += self._HSIC(K, L)
                    hsic_matrix[cnt, 0] += self._HSIC(K, K)
                    hsic_matrix[cnt, 2] += self._HSIC(L, L)
                    cnt += 1

        hsic_matrix = hsic_matrix[:, 1] / (hsic_matrix[:, 0].sqrt() * hsic_matrix[:, 2].sqrt())
        loss = torch.sum(hsic_matrix)

        return loss