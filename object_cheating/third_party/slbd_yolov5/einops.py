import torch


def rearrange(x, pattern: str, **sizes):
    if pattern == "b n (h c) -> (b h) n c":
        h = sizes["h"]
        b, n, hc = x.shape
        c = hc // h
        return x.reshape(b, n, h, c).permute(0, 2, 1, 3).reshape(b * h, n, c)

    if pattern == "b (w h) c -> b c w h":
        w = sizes["w"]
        h = sizes["h"]
        b, wh, c = x.shape
        if wh != w * h:
            raise ValueError(f"Expected second dimension {w*h}, got {wh}")
        return x.reshape(b, w, h, c).permute(0, 3, 1, 2)

    if pattern == "b c w h -> b (w h) c":
        b, c, w, h = x.shape
        return x.permute(0, 2, 3, 1).reshape(b, w * h, c)

    if pattern == "(b h) n c -> b n (h c)":
        h = sizes["h"]
        bh, n, c = x.shape
        if bh % h != 0:
            raise ValueError(f"First dimension {bh} is not divisible by h={h}")
        b = bh // h
        return x.reshape(b, h, n, c).permute(0, 2, 1, 3).reshape(b, n, h * c)

    raise NotImplementedError(f"Unsupported rearrange pattern: {pattern}")
