import torch


class GeometryPrior(torch.nn.Module):
    def __init__(self, k, channels, multiplier=0.5):
        super(GeometryPrior, self).__init__()
        self.channels = channels
        self.k = k
        self.position = 2 * torch.rand(1, 2, k, k, requires_grad=True) - 1
        self.l1 = torch.nn.Conv2d(2, int(multiplier * channels), 1)
        self.l2 = torch.nn.Conv2d(int(multiplier * channels), channels, 1)

    def forward(self, x):
        x = self.l2(torch.nn.functional.relu(self.l1(self.position)))
        return x.view(1, self.channels, 1, self.k ** 2)



class KeyQueryMap(torch.nn.Module):
    def __init__(self, channels, m):
        super(KeyQueryMap, self).__init__()
        self.l = torch.nn.Conv2d(channels, channels // m, 1)

    def forward(self, x):
        return self.l(x)


class AppearanceComposability(torch.nn.Module):
    def __init__(self, k, padding, stride):
        super(AppearanceComposability, self).__init__()
        self.k = k
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)

    def forward(self, x):
        key_map, query_map = x
        k = self.k
        key_map_unfold = self.unfold(key_map)
        query_map_unfold = self.unfold(query_map)
        key_map_unfold = key_map_unfold.view(
            key_map.shape[0], key_map.shape[1],
            -1,
            key_map_unfold.shape[-2] // key_map.shape[1])
        query_map_unfold = query_map_unfold.view(
            query_map.shape[0], query_map.shape[1],
            -1,
            query_map_unfold.shape[-2] // query_map.shape[1])
        return key_map_unfold * query_map_unfold[:, :, :, k**2//2:k**2//2+1]


def combine_prior(appearance_kernel, geometry_kernel):
    return torch.nn.functional.softmax(appearance_kernel + geometry_kernel,
                                       dim=-1)


class LocalRelationalLayer(torch.nn.Module):
    def __init__(self, channels, k=7, stride=1, m=8, padding=3):
        super(LocalRelationalLayer, self).__init__()
        self.channels = channels
        self.k = k
        self.stride = stride
        self.m = m
        self.padding = padding
        self.kmap = KeyQueryMap(channels, self.m)
        self.qmap = KeyQueryMap(channels, self.m)
        self.ac = AppearanceComposability(k, padding, stride)
        self.gp = GeometryPrior(k, channels//self.m)
        self.unfold = torch.nn.Unfold(k, 1, padding, stride)
        self.final1x1 = torch.nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        gpk = self.gp(0)
        km = self.kmap(x)
        qm = self.qmap(x)
        ak = self.ac((km, qm))
        ck = combine_prior(ak, gpk)[:, None, :, :, :]
        x_unfold = self.unfold(x)
        x_unfold = x_unfold.view(x.shape[0], self.m, x.shape[1] // self.m,
                                 -1, x_unfold.shape[-2] // x.shape[1])
        pre_output = (ck * x_unfold)
        h_out = (x.shape[2] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                self.stride + 1
        w_out = (x.shape[3] + 2 * self.padding - 1 * (self.k - 1) - 1) // \
                self.stride + 1
        pre_output = torch.sum(pre_output, axis=-1).view(x.shape[0], x.shape[1],
                                                         h_out, w_out)
        return self.final1x1(pre_output)


if __name__ == '__main__':
    x = torch.ones(2, 32, 64, 64)
    lrn = LocalRelationalLayer(32, k=7, padding=3)
    o = lrn(x)
    print(o.shape)
