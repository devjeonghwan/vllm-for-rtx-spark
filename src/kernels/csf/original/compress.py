# INERT STATIC RECONSTRUCTION. Do not import or execute this artifact.
# The Cython-generated AArch64 instruction stream is the authority, not these
# reconstructed variable names or formatting. No original source text exists.
# Original ELF SHA256:
# 94bd4612f7b3a2bc07b6dc51e99b312c9616a1960cceee45d5bbf2b66b56510d
# Module-state constants: 0x6e0=0, 0x6e8=-1, 0x6f0=1, 0x6f8=2,
# 0x708=4, 0x710=15, 0x718=16, 0x720=255 (0x5528..0x5574).
# Constant tuples and slices: 0x6124..0x62d8.

def compress(scales):
    # 0x16ff0..0x17220: contiguous, uint8 view, (-1,16) reshape, int16 cast.
    x = scales.contiguous().view(torch.uint8).reshape(-1, 16).to(torch.int16)
    # 0x17274..0x172d4 and 0x17304..0x17360.
    mn = x.min(1).values
    # Subtract helper 0xb660 -> PyNumber_Subtract; GT opcode4 at 0x18a74;
    # EQ opcode2 at 0x190a8; OR helper 0xb434 -> PyNumber_Or.
    raw = ((x.max(1).values - mn) > 15) | (mn == 255)
    # 0x177b4..0x17948: [:,None], subtract, clamp(0,15), uint8 cast.
    off = (x - mn[:, None]).clamp(0, 15).to(torch.uint8)
    # 0x17990..0x17ae8: even/odd slices, shift4, OR, contiguous.
    packed = (off[:, 0::2] | (off[:, 1::2] << 4)).contiguous()
    # 0x17b2c..0x17ca4: int32 view, (-1,2) reshape, clone.
    nib = packed.view(torch.int32).reshape(-1, 2).clone()
    # 0x17ce8..0x17dc4; only the function invocation uses the tuple/call helper.
    ridx = torch.nonzero(raw).flatten()
    # 0x17e08..0x18068: arange dtype and device explicitly from x.device.
    nib[ridx, 0] = torch.arange(ridx.numel(), device=x.device, dtype=torch.int32)
    # 0x18094..0x18110.
    nib[ridx, 1] = 0
    # 0x18154..0x183d4: where(raw, full_like(mn,255), mn), uint8 cast.
    base = torch.where(raw, torch.full_like(mn, 255), mn).to(torch.uint8)
    # 0x183dc..0x186cc: gather, uint8 cast, contiguous, int32 view, (-1,4).
    rawv = x[ridx].to(torch.uint8).contiguous().view(torch.int32).reshape(-1, 4)
    # 0x186e0..0x18750, 0x19e30..0x19fac: empty raw plane has one zero row.
    if rawv.numel() == 0:
        rawv = torch.zeros((1, 4), dtype=torch.int32, device=x.device)
    # 0x18754..0x18850: three contiguous calls, then a tuple of three tensors.
    return base.contiguous(), nib.contiguous(), rawv.contiguous()
