import os
import math
from random import shuffle
from time import time
import cupy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
import torchvision.transforms as transforms


@dataclass
class Config:
    name: str
    embed_dim: int
    depth: int
    num_heads: int

class QWeight:
    def __init__(self, weights, name):
        self.scale = weights[f"{name}_scale"]
    def get_reshape_range(self, inputs):
        if len(inputs.shape) == 2:
            range_shape = (-1, 1)
        elif len(inputs.shape) == 4:
            range_shape = (-1, 1, 1, 1)
        else:
            raise NotImplementedError
        return range_shape
    def quant(self, inputs):
        assert inputs.dtype == np.float32, inputs.dtype
        range_shape = self.get_reshape_range(inputs)
        scale = self.scale.reshape(range_shape)
        outputs = inputs / scale
        outputs = np.clip(outputs.round(), -2**7, 2**7 - 1)
        return outputs.astype(np.int8)
    def dequant(self, inputs):
        assert inputs.dtype == np.int8, inputs.dtype
        range_shape = self.get_reshape_range(inputs)
        scale = self.scale.reshape(range_shape)
        outputs = inputs.astype(np.float32)
        outputs = outputs * scale
        return outputs.astype(np.float32)

class QAct:
    def __init__(self, weights, name):
        self.scale = weights[f"{name}_scale"]
    def get_reshape_range(self, inputs):
        if len(inputs.shape) == 2:
            range_shape = (1, -1)
        elif len(inputs.shape) == 3:
            range_shape = (1, 1, -1)
        elif len(inputs.shape) == 4:
            range_shape = (1, -1, 1, 1)
        else:
            raise NotImplementedError
        return range_shape
    def quant(self, inputs):
        assert inputs.dtype == np.float32, inputs.dtype
        range_shape = self.get_reshape_range(inputs)
        scale = self.scale.reshape(range_shape)
        outputs = inputs / scale
        outputs = np.clip(outputs.round(), -2**7, 2**7 - 1)
        return outputs.astype(np.int8)
    def dequant(self, inputs):
        assert inputs.dtype == np.int8, inputs.dtype
        range_shape = self.get_reshape_range(inputs)
        scale = self.scale.reshape(range_shape)
        outputs = inputs.astype(np.float32)
        outputs = outputs * scale
        return outputs.astype(np.float32)

def int_matmul(x, w, b, xq: QAct, wq: QWeight, zq: QAct, act=None):
    assert x.dtype == np.int8, x.dtype
    assert w.dtype == np.int8, w.dtype
    N, D = x.shape
    w = w.transpose(1, 0)   # (D, E)
    # We need int8 multiplication with int32 accumulation. cupy does not support this.
    if np.__name__ == 'numpy':
        xw = np.matmul(x, w, dtype=np.int32)  # (N, E)
    else:
        xw = np.matmul(x.astype(np.int32), w.astype(np.int32), dtype=np.int32)  # (N, E)
    xw = xw + b / (xq.scale * wq.scale)
    xw = xw * (xq.scale * wq.scale / zq.scale)
    xw = np.clip(xw.round(), -2**7, 2**7 - 1)
    xw = xw.astype(np.int8)
    if act is not None:
        xw = act(xw, zq)
    return xw

class QConv2d:
    def __init__(self, config: Config, weights, name):
        self.config = config
        self.weight = weights[f"{name}_weight"]
        self.bias = weights[f"{name}_bias"]
        assert self.bias.shape == (config.embed_dim,)
        self.bias = self.bias.reshape(1, config.embed_dim)
        self.quantizer = QWeight(weights, name)
        self.weight = self.quantizer.quant(self.weight)
        assert self.weight.shape == (config.embed_dim, 3, 16, 16)
        self.weight = self.weight.reshape(config.embed_dim, 768)
    def __call__(self, x, xq: QAct, zq: QAct):
        assert x.dtype == np.int8, x.dtype
        assert x.shape == (1, 3, 224, 224)
        reshaped_x = np.zeros((768, 14 * 14), dtype=np.int8)
        for i in range(14):
            for j in range(14):
                reshaped_x[:, i * 14 + j] = x[0, :, i*16:i*16+16, j*16:j*16+16].flatten()
        output = int_matmul(reshaped_x.T, self.weight, self.bias, xq, self.quantizer, zq).T
        output = output.reshape(1, self.config.embed_dim, 14 * 14).transpose(0, 2, 1)
        return output

class QLinear:
    def __init__(self, weights, name):
        self.weight = weights[f"{name}_weight"]
        self.bias = weights[f"{name}_bias"]
        self.quantizer = QWeight(weights, name)
        self.weight = self.quantizer.quant(self.weight)
    def __call__(self, x, xq: QAct, zq: QAct, act=None):
        assert x.dtype == np.int8, x.dtype
        if x.ndim == 3:  # B, N, D
            assert x.shape[0] == 1, x.shape
            x = int_matmul(x[0], self.weight, self.bias.reshape(1, -1), xq, self.quantizer, zq, act)
            x = np.expand_dims(x, 0)
        else:   # B, D
            x = int_matmul(x, self.weight, self.bias.reshape(1, 1, -1), xq, self.quantizer, zq, act)
        return x

class QIntLayerNorm:
    def __init__(self, weights, name):
        self.weight = weights[f"{name}_weight"]
        self.bias = weights[f"{name}_bias"]
    def __call__(self, x, xq: QAct, zq: QAct):
        assert x.dtype == np.int8, x.dtype
        in_scale = xq.scale
        channel_nums = x.shape[-1]
        in_scale = in_scale.reshape(1, 1, -1)
        in_scale1 = in_scale.min()
        in_scale_mask = np.log2((in_scale / in_scale1).round()).astype(np.int8)
        assert (in_scale_mask <= 8).all(), in_scale_mask.max()
        x_q = np.left_shift(x, in_scale_mask, dtype=np.int16)
        sum_x_q = x_q.sum(axis=-1, dtype=np.int32)
        var_x_q = channel_nums * np.multiply(x_q, x_q, dtype=np.int32).sum(axis=-1) - sum_x_q**2
        inv_std_x_q = 1.0 / np.sqrt(var_x_q.astype(np.float32))
        out_scale = zq.scale.reshape(1, 1, -1)
        weight = self.weight.reshape(1, 1, -1) / out_scale
        bias = self.bias.reshape(1, 1, -1) / out_scale
        x = (x_q.astype(np.float32) * np.expand_dims(channel_nums * inv_std_x_q, -1)
             - np.expand_dims(sum_x_q.astype(np.float32) * inv_std_x_q, -1)
            ) * weight + bias
        x = np.clip(x.round(), -2**7, 2**7 - 1)
        return x.astype(np.int8)

class QIntSoftmax:
    def __call__(self, x, xq: QAct):
        assert x.dtype == np.int8, x.dtype
        x = x.astype(np.int16)
        x = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x.astype(np.float32) * xq.scale)
        ex_sum = ex.sum(axis=-1, keepdims=True)
        x = ex / ex_sum * 255
        x = np.clip(x.round(), 0, 255)
        return x.astype(np.uint8)

def mm_attention(q, k, inq: QAct, out_scale, outq: QAct):
    assert q.dtype == np.int8, q.dtype
    assert k.dtype == np.int8, k.dtype
    k = k.transpose(0, 1, 3, 2)
    z = np.einsum('BHMN,BHNK->BHMK', q, k, dtype=np.int32)
    z = z * (inq.scale * inq.scale * out_scale / outq.scale)
    z = np.clip(z.round(), -2**7, 2**7 - 1)
    z = z.astype(np.int8)
    return z

class Attention:
    def __init__(self, config: Config, weights, name):
        self.config = config
        self.qkv = QLinear(weights, f"{name}.qkv")
        self.qact1 = QAct(weights, f"{name}.qact1")
        head_dim = config.embed_dim // config.num_heads
        self.scale = head_dim**-0.5
        self.qact_attn1 = QAct(weights, f"{name}.qact_attn1")
        self.log_int_softmax = QIntSoftmax()
        self.qact2 = QAct(weights, f"{name}.qact2")
        self.proj = QLinear(weights, f"{name}.proj")
        self.qact3 = QAct(weights, f"{name}.qact3")
    def __call__(self, x, xq: QAct):
        assert x.dtype == np.int8, x.dtype
        B, N, C = x.shape
        x = self.qkv(x, xq, self.qact1)
        qkv = x.reshape(B, N, 3, self.config.num_heads,
                        C // self.config.num_heads).transpose(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = mm_attention(q, k, self.qact1, self.scale, self.qact_attn1)
        attn = self.log_int_softmax(attn, self.qact_attn1)
        x = np.einsum('BHMN,BHNK->BHMK', attn, v, dtype=np.int32)
        x = x * (self.qact1.scale / self.qact2.scale / 255)
        x = np.clip(x.round(), -2**7, 2**7 - 1).astype(np.int8)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x, self.qact2, self.qact3)
        return x, self.qact3

class GELU:
    def __init__(self):
        self.break_points = [-2.424, -0.623, 0, 0.623, 2.424]
        self.slopes = [-0.1, 0.26438, 0.73562, 1.1]
        self.offset = [-0.2424, -0.01539, -0.01539, -0.2424]
    def __call__(self, x, xq: QAct):
        assert x.dtype == np.int8, x.dtype
        x = np.select(
            [x < bp / xq.scale for bp in self.break_points] + [np.ones_like(x) > 0],
            [np.zeros_like(x)] + [x.astype(np.float32) * s + o / xq.scale for s, o in zip(self.slopes, self.offset)] + [x])
        x = np.clip(x.round(), -2**7, 2**7 - 1)
        x = x.astype(np.int8)
        return x

class Mlp:
    def __init__(self, weights, name):
        self.fc1 = QLinear(weights, f"{name}.fc1")
        self.act = GELU()
        self.qact1 = QAct(weights, f"{name}.qact1")
        self.fc2 = QLinear(weights, f"{name}.fc2")
        self.qact2 = QAct(weights, f"{name}.qact2")
    def __call__(self, x, xq):
        x = self.fc1(x, xq, self.qact1, act=self.act)
        x = self.fc2(x, self.qact1, self.qact2)
        return x, self.qact2

def q_add(a, b, aq: QAct, bq: QAct, cq: QAct):
    assert a.dtype == np.int8, a.dtype
    assert b.dtype == np.int8, b.dtype
    s_a2c = aq.scale / cq.scale
    s_b2c = bq.scale / cq.scale
    c = s_a2c * a + s_b2c * b
    c = c.round().clip(-2**7, 2**7 - 1)
    c = c.astype(np.int8)
    return c, cq

class Block:
    def __init__(self, config, weights, name):
        self.norm1 = QIntLayerNorm(weights, f"{name}.norm1")
        self.qact1 = QAct(weights, f"{name}.qact1")
        self.attn = Attention(config, weights, f"{name}.attn")
        self.qact2 = QAct(weights, f"{name}.qact2")
        self.norm2 = QIntLayerNorm(weights, f"{name}.norm2")
        self.qact3 = QAct(weights, f"{name}.qact3")
        self.mlp = Mlp(weights, f"{name}.mlp")
        self.qact4 = QAct(weights, f"{name}.qact4")
    def __call__(self, x, xq: QAct):
        assert x.dtype == np.int8, x.dtype
        y, yq = self.attn(self.norm1(x, xq, self.qact1), self.qact1)
        x, xq = q_add(x, y, xq, yq, self.qact2)
        y, yq = self.mlp(self.norm2(x, self.qact2, self.qact3), self.qact3)
        x, xq = q_add(x, y, xq, yq, self.qact4)
        return x, xq

class DeiT:
    def __init__(self, config: Config):
        print("Loading Model:", config.name)
        self.config = config
        weights = np.load(f"export/{config.name}.npz")
        if config.name.split('_')[0] != 'vit':
            self.qact_input = QAct(weights, "qact_input")
        else:
            self.qact_input = QAct({"fake_scale": np.array([1.0 / 127])}, "fake")
        self.patch_embed_proj = QConv2d(config, weights, "patch_embed.proj")
        self.patch_embed_qact = QAct(weights, "patch_embed.qact")
        self.cls_token = weights["cls_token"]
        self.qact_embed = QAct(weights, "qact_embed")
        self.pos_embed = weights["pos_embed"]
        self.qact1 = QAct(weights, "qact1")
        self.cls_token = self.qact1.quant(self.cls_token)
        self.pos_embed = (self.pos_embed / self.qact1.scale)#.astype(np.int32)
        self.blocks = [Block(config, weights, f"blocks.{i}") for i in range(config.depth)]
        self.norm = QIntLayerNorm(weights, "norm")
        self.qact2 = QAct(weights, "qact2")
        self.head = QLinear(weights, "head")
        self.act_out = QAct(weights, "act_out")
    def __call__(self, x):
        x = self.qact_input.quant(x)
        x = self.patch_embed_proj(x, self.qact_input, self.qact1)
        x = np.concatenate((self.cls_token, x), axis=1)
        x = (x + self.pos_embed).astype(np.int8)
        xq = self.qact1
        for b in self.blocks:
            x, xq = b(x, xq)
        x = self.norm(x, xq, self.qact2)[:, 0]
        x = self.head(x, self.qact2, self.act_out)
        return x

def build_transform(name):
    model_type = name.split('_')[0]
    if model_type == 'deit':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif model_type == 'vit':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif model_type == 'swin':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError
    t = []
    input_size=224
    size = int(math.floor(input_size / crop_pct))

    t.append(transforms.Resize(size, interpolation=Image.BICUBIC))
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


deit_tiny = Config('deit_tiny', 192, 12, 3)
deit_small = Config('deit_small', 384, 12, 6)
deit_base = Config('deit_base', 768, 12, 12)
vit_base = Config('vit_base', 768, 12, 12)
vit_large = Config('vit_large', 1024, 24, 16)

config = deit_tiny
valdir = '/media/dinger/inner/Dataset/ImageNet/val'
with open(os.path.join(valdir, "val_list.txt"), "r") as f:
    lines = f.readlines()
shuffle(lines)
top1, top5 = 0, 0
model = DeiT(config)
transform = build_transform(config.name)
times = []
pbar = tqdm(lines)
for i, line in enumerate(pbar):
    now_img_idx = i
    name, label = line.strip().split()
    label = int(label)
    path = os.path.join(valdir, name)
    img = Image.open(path)
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0)
    img = np.asarray(img.numpy())
    start = time()
    output = model(img)
    times.append(time() - start)
    pred = np.argsort(output[0][0])[-5:][::-1]
    top1 += (pred[0] == label)
    top5 += (label in pred)
    pbar.set_description(f"top1={100.*top1/(i+1):.2f}%, top5={100.*top5/(i+1):.2f}%")

times = np.array(times)
print(f"evaluated {len(lines)} images, time: avg {np.mean(times):.4f}s, min {np.min(times):.4f}s, max {np.max(times):.4f}s")
print(f"top1 = {top1}, top5 = {top5}")
