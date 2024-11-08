import os
import math
from time import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
crop_pct = 0.875
t = []
input_size=224
size = int(math.floor(input_size / crop_pct))

t.append(transforms.Resize(size, interpolation=Image.BICUBIC))
t.append(transforms.CenterCrop(input_size))
t.append(transforms.ToTensor())
t.append(transforms.Normalize(mean, std))
transform = transforms.Compose(t)

class QAct:
    def __init__(self, name, module_type='activation'):
        self.module_type = module_type
        self.scale = np.load(f"export/{name}_scale.npy")
        self.zero_point = np.load(f"export/{name}_zero_point.npy")
    def get_reshape_range(self, inputs):
        if self.module_type == 'weight':
            if len(inputs.shape) == 2:
                range_shape = (-1, 1)
            elif len(inputs.shape) == 4:
                range_shape = (-1, 1, 1, 1)
            else:
                raise NotImplementedError
        else:
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
        zero_point = self.zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        # return outputs.astype(np.uint8)
        if self.module_type == 'activation':
            outputs = np.clip(outputs.round(), 0, 2**8 - 1)
            return outputs.astype(np.uint8)
        elif self.module_type == 'weight':
            outputs = np.clip(outputs.round(), -2**7, 2**7 - 1)
            return outputs.astype(np.int8)
        else:
            raise NotImplementedError
    def dequant(self, inputs):
        if self.module_type == 'activation':
            assert inputs.dtype == np.uint8, inputs.dtype
        else:
            assert inputs.dtype == np.int8, inputs.dtype
        range_shape = self.get_reshape_range(inputs)
        scale = self.scale.reshape(range_shape)
        zero_point = self.zero_point.reshape(range_shape)
        outputs = (inputs.astype(np.float32) - zero_point) * scale
        return outputs.astype(np.float32)

def int_matmul(x, w, xq: QAct, wq: QAct):   # zq: QAct
    assert x.dtype == np.uint8, x.dtype
    assert w.dtype == np.int8, w.dtype
    N, D = x.shape
    w = w.transpose(1, 0)   # (D, E)
    xw = np.matmul(x, w, dtype=np.int32)  # (N, E)
    wz = wq.zero_point.astype(np.float32)
    xz = xq.zero_point.astype(np.float32)
    x = x.astype(np.float32)
    w = w.astype(np.float32)
    xw = xw.astype(np.float32)
    xw = xq.scale * wq.scale * (xw - x.sum(axis=1, keepdims=True) * wz \
                                   - w.sum(axis=0, keepdims=True) * xz \
                                   + (wz * xz * D))
    assert xw.dtype == np.float32, xw.dtype
    return xw

class QConv2d:
    def __init__(self, name):
        self.weight = np.load(f"export/{name}_weight.npy")
        self.bias = np.load(f"export/{name}_bias.npy")
        self.quantizer = QAct(name, 'weight')
        self.weight = self.quantizer.quant(self.weight)
    def __call__(self, x, xq):
        assert x.dtype == np.uint8, x.dtype
        assert x.shape == (1, 3, 224, 224)
        assert self.weight.shape == (192, 3, 16, 16)
        assert self.bias.shape == (192,)
        reshaped_x = np.zeros((768, 14 * 14), dtype=np.uint8)
        for i in range(14):
            for j in range(14):
                reshaped_x[:, i * 14 + j] = x[0, :, i*16:i*16+16, j*16:j*16+16].flatten()
        weight = self.weight.reshape(192, 768)
        output = int_matmul(reshaped_x.T, weight, xq, self.quantizer).T
        output = output + self.bias.reshape(192, 1)
        return output.reshape(1, 192, 14 * 14).transpose(0, 2, 1)

class QLinear:
    def __init__(self, name):
        self.weight = np.load(f"export/{name}_weight.npy")
        self.bias = np.load(f"export/{name}_bias.npy")
        self.quantizer = QAct(name, 'weight')
        self.weight = self.quantizer.quant(self.weight)
    def __call__(self, x, xq):
        assert x.dtype == np.uint8, x.dtype
        if x.ndim == 3:  # B, N, D
            assert x.shape[0] == 1, x.shape
            x = int_matmul(x[0], self.weight, xq, self.quantizer)
            x = np.expand_dims(x, 0)
        else:   # B, D
            x = int_matmul(x, self.weight, xq, self.quantizer)
        x = x + self.bias.reshape(1, 1, -1)
        return x

class QIntLayerNorm:
    def __init__(self, name):
        self.weight = np.load(f"export/{name}_weight.npy")
        self.bias = np.load(f"export/{name}_bias.npy")
    def get_MN(self, x):
        bit = 8
        N = np.clip(bit - 1 - np.floor(np.log2(x)), 0, 31)
        M = np.clip(np.floor(x * np.power(2, N)), 0, 2 ** bit - 1)
        return M, N
    def __call__(self, x, in_quantizer, out_quantizer):
        assert x.dtype == np.uint8, x.dtype
        in_scale = in_quantizer.scale
        out_scale = out_quantizer.scale
        channel_nums = x.shape[-1]
        in_scale = in_scale.reshape(1, 1, -1)
        out_scale = out_scale.reshape(1, 1, -1)
        x_q = x.astype(np.float32) - in_quantizer.zero_point
        in_scale1 = in_scale.min()
        in_scale_mask = (in_scale / in_scale1).round()
        x_q = x_q * in_scale_mask
        mean_x_q = x_q.mean(axis=-1) * in_scale1
        std_x_q = (in_scale1 / channel_nums) * np.sqrt(
            channel_nums * (x_q**2).sum(axis=-1) - x_q.sum(axis=-1)**2)
        A = np.expand_dims(in_scale1 / std_x_q, -1) * \
            self.weight.reshape(1, 1, -1) / out_scale
        A_sign = np.sign(A)
        M, N = self.get_MN(np.abs(A))
        B = ((self.bias.reshape(1, 1, -1) -
                np.expand_dims(mean_x_q / std_x_q, -1) *
                self.weight.reshape(1, 1, -1)) / out_scale *
                np.power(2, N)).round()
        x_q = ((A_sign * M * x_q + B) / np.power(2, N)).round()
        x = x_q * out_scale
        return x

class QIntSoftmax:
    @staticmethod
    def log_round(x):
        x_log_floor = np.floor(np.log2(x))
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big
    @staticmethod
    def int_softmax(x, xq: QAct):
        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = np.floor(coef[1] / scaling_factor)
            c_int = np.floor(coef[2] / scaling_factor**2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor**2
            return z, scaling_factor
        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = np.floor(x0 / scaling_factor)
            x_int = np.maximum(x_int, n * x0_int)
            q = np.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = np.clip(np.floor(exp_int * 2**(n - q)), 0, None)
            scaling_factor = exp_scaling_factor / 2**n
            return exp_int, scaling_factor
        x_int = x.astype(np.float32) - xq.zero_point
        x_int_max = x_int.max(axis=-1, keepdims=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, xq.scale)
        exp_int_sum = exp_int.sum(axis=-1, keepdims=True)
        return exp_int, exp_int_sum
    def __call__(self, x, xq):
        assert x.dtype == np.uint8, x.dtype
        exp_int, exp_int_sum = self.int_softmax(x, xq)
        softmax_out = np.round(exp_int_sum / exp_int)
        rounds = self.log_round(softmax_out)    # uint4
        mask = rounds >= 2**4
        qlog = np.clip(rounds, 0, 2**4 - 1)
        deq_softmax = 2**(-qlog)
        deq_softmax[mask] = 0
        return deq_softmax.astype(np.float32)

def mm_attention(q, k, qact):
    assert q.dtype == np.uint8, q.dtype
    assert k.dtype == np.uint8, k.dtype
    k = k.transpose(0, 1, 3, 2)
    z = np.einsum('BHMN,BHNK->BHMK', q, k, dtype=np.uint32)
    qz = qact.zero_point.astype(np.float32)
    z = z.astype(np.float32)
    qkz = q.astype(np.float32).sum(axis=3, keepdims=True) * qz
    qzk = k.astype(np.float32).sum(axis=2, keepdims=True) * qz
    qzkz = qz * qz * q.shape[-1]
    z = qact.scale * qact.scale * (z - qkz - qzk + qzkz)
    assert z.dtype == np.float32, z.dtype
    return z

class Attention:
    def __init__(self, name):
        dim = 192
        self.num_heads = 3
        self.qkv = QLinear(f"{name}.qkv")
        self.qact1 = QAct(f"{name}.qact1")
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qact_attn1 = QAct(f"{name}.qact_attn1")
        self.log_int_softmax = QIntSoftmax()
        self.qact2 = QAct(f"{name}.qact2")
        self.proj = QLinear(f"{name}.proj")
        self.qact3 = QAct(f"{name}.qact3")
    def __call__(self, x, xq):
        assert x.dtype == np.uint8, x.dtype
        B, N, C = x.shape
        x = self.qkv(x, xq)
        x = self.qact1.quant(x)
        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).transpose(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = qkv[0], qkv[1], qkv[2]
        v = self.qact1.dequant(v)
        attn = mm_attention(q, k, self.qact1) * self.scale
        attn = self.qact_attn1.quant(attn)
        attn = self.log_int_softmax(attn, self.qact_attn1)
        x = np.einsum('BHMN,BHNK->BHMK', attn, v)
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.qact2.quant(x)
        x = self.proj(x, self.qact2)
        x = self.qact3.quant(x)
        return x, self.qact3

class GELU:
    def __call__(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

class Mlp:
    def __init__(self, name):
        self.fc1 = QLinear(f"{name}.fc1")
        self.act = GELU()
        self.qact1 = QAct(f"{name}.qact1")
        self.fc2 = QLinear(f"{name}.fc2")
        self.qact2 = QAct(f"{name}.qact2")
    def __call__(self, x, xq):
        x = self.fc1(x, xq)
        x = self.act(x)
        x = self.qact1.quant(x)
        x = self.fc2(x, self.qact1)
        x = self.qact2.quant(x)
        return x, self.qact2

def q_add(a, b, aq: QAct, bq: QAct, zq: QAct):
    assert a.dtype == np.uint8, a.dtype
    assert b.dtype == np.uint8, b.dtype
    z = aq.dequant(a) + bq.dequant(b)
    return zq.quant(z), zq

class Block:
    def __init__(self, name):
        self.norm1 = QIntLayerNorm(f"{name}.norm1")
        self.qact1 = QAct(f"{name}.qact1")
        self.attn = Attention(f"{name}.attn")
        self.qact2 = QAct(f"{name}.qact2")
        self.norm2 = QIntLayerNorm(f"{name}.norm2")
        self.qact3 = QAct(f"{name}.qact3")
        self.mlp = Mlp(f"{name}.mlp")
        self.qact4 = QAct(f"{name}.qact4")
    def __call__(self, x, xq: QAct):
        assert x.dtype == np.uint8, x.dtype
        y, yq = self.attn(self.qact1.quant(self.norm1(x, xq, self.qact1)), self.qact1)
        x, xq = q_add(x, y, xq, yq, self.qact2)
        y, yq = self.mlp(self.qact3.quant(self.norm2(x, self.qact2, self.qact3)), self.qact3)
        x, xq = q_add(x, y, xq, yq, self.qact4)
        return x, xq

class DeiT_tiny:
    def __init__(self):
        self.qact_input = QAct("qact_input")
        self.patch_embed_proj = QConv2d("patch_embed.proj")
        self.patch_embed_qact = QAct("patch_embed.qact")
        self.cls_token = np.load("export/cls_token.npy")
        self.qact_embed = QAct("qact_embed")
        self.pos_embed = np.load("export/pos_embed.npy")
        self.qact1 = QAct("qact1")
        self.cls_token = self.qact_embed.dequant(self.qact_embed.quant(self.cls_token))
        self.cls_token = self.qact1.quant(self.cls_token)
        self.pos_embed = (self.pos_embed / self.qact1.scale)#.astype(np.int32)
        self.blocks = [Block(f"blocks.{i}") for i in range(12)]
        self.norm = QIntLayerNorm("norm")
        self.qact2 = QAct("qact2")
        self.head = QLinear("head")
        self.act_out = QAct("act_out")
    def __call__(self, x):
        x = self.qact_input.quant(x)
        x = self.patch_embed_proj(x, self.qact_input)
        x = self.patch_embed_qact.dequant(self.patch_embed_qact.quant(x))
        x = self.qact_embed.dequant(self.qact_embed.quant(x))
        x = self.qact1.quant(x)
        x = np.concatenate((self.cls_token, x), axis=1)
        x = (x + self.pos_embed).astype(np.uint8)
        xq = self.qact1
        for b in self.blocks:
            x, xq = b(x, xq)
        x = self.norm(x, xq, self.qact2)[:, 0]
        x = self.qact2.quant(x)
        x = self.head(x, self.qact2)
        x = self.act_out.quant(x)
        return x

valdir = '/media/dinger/inner/Dataset/ImageNet/val'
with open(os.path.join(valdir, "val_list.txt"), "r") as f:
    lines = f.readlines()
top1, top5 = 0, 0
model = DeiT_tiny()
times = []
for i, line in enumerate(tqdm(lines)):
    now_img_idx = i
    name, label = line.strip().split()
    label = int(label)
    path = os.path.join(valdir, name)
    img = Image.open(path)
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0)
    img = img.numpy()
    start = time()
    output = model(img)
    times.append(time() - start)
    pred = np.argsort(output[0][0])[-5:][::-1]
    top1 += (pred[0] == label)
    top5 += (label in pred)

print(f"evaluated {len(lines)} images, time: avg {np.mean(times):.4f}s, min {np.min(times):.4f}s, max {np.max(times):.4f}s")
print(f"top1 = {top1}, top5 = {top5}")
