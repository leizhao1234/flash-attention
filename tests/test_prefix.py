import time
import torch
from torch.nn import functional as F
from flash_attn import flash_attn_func_prefix
from einops import rearrange
import math

def standard_attention(query_layer, key_layer, value_layer, attention_mask,scaling_attention_score=True):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training.
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers.
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = torch.mul(attention_scores, attention_mask) - \
                           10000.0 * (1.0 - attention_mask)

    attention_probs = F.softmax(attention_scores, dim=-1)


    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer


def flash_attention(q, k, v, prefix=True, prefix_lens=None):
    o = flash_attn_func_prefix(q, k, v, prefix=prefix, prefix_lens=prefix_lens)
    return o


def test(func_name, q, k, v, *args, **kwargs):
    if func_name in ["standard_attention", "pytorch_func"]:
        q = rearrange(q, "a b c d -> a c b d")
        k = rearrange(k, "a b c d -> a c b d")
        v = rearrange(v, "a b c d -> a c b d")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    for _ in range(5):
        o = globals()[func_name](q, k, v, *args, **kwargs)
    torch.cuda.synchronize()
    st = time.time()
    o = globals()[func_name](q, k, v, *args, **kwargs)
    torch.cuda.synchronize()
    tt = time.time() - st
    max_memory = torch.cuda.max_memory_allocated() // 2**20
    torch.cuda.empty_cache()
    print("o shape is ", o.size())
    if func_name in ["standard_attention", "pytorch_func"]:
        o = rearrange(o, "a c b d -> a b c d")

    return o, tt, max_memory

if __name__ == "__main__":
    batch_size = 20
    seq_len = 1025
    num_head = 32
    hidden_units = 128
    torch.manual_seed(0)
    prefix_lens = torch.tensor([[100,150,200,201,302,405,507,608,711,899,100,150,200,201,302,405,507,608,711,899,100,150,200,201,302,405,507,608,711,899]], dtype=torch.int32,device="cuda")
    #attn_mask = torch.ones(3, num_head, seq_len, seq_len, dtype=torch.bool,device="cuda").tril(diagonal=0)
    #attn_mask[..., :prefix_len] = True
    #print(attn_mask)
    #print(attn_mask.size())
    f_attn_mask = torch.ones(batch_size, num_head, seq_len, seq_len, dtype=torch.float16, device="cuda")
    f_attn_mask.tril_()
    for i in range(batch_size):
        f_attn_mask[i,...,:prefix_lens[0,i]] = 1

    #f_attn_mask[..., :prefix_len] = 1
    print(f_attn_mask)
    test_num = 10
    for idx in range(test_num):
        print(f"test {idx} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        query = torch.rand((batch_size, seq_len, num_head, hidden_units), dtype=torch.float16, device="cuda")
        key = torch.rand((batch_size, seq_len, num_head, hidden_units), dtype=torch.float16, device="cuda")
        value =  torch.rand((batch_size, seq_len, num_head, hidden_units), dtype=torch.float16, device="cuda")
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        o_ref,_,_ = test("standard_attention", query.float(), key.float(), value.float(), attention_mask=f_attn_mask.float(), scaling_attention_score=True)
        o_ref.to(dtype=query.dtype)
        g = torch.randn_like(o_ref)
        o_ref.backward(g)

        q_py = query.detach().clone().requires_grad_(True)
        k_py = key.detach().clone().requires_grad_(True)
        v_py = value.detach().clone().requires_grad_(True)
        o, t, m = test("standard_attention", q_py, k_py, v_py, attention_mask=f_attn_mask, scaling_attention_score=True)
        print("o shape is ", o.size())
        print(f"custom pytorch time: {t:.6f}, peak memory: {m} MB")
        #print(o)
        o.backward(g)

        q_fa = query.detach().clone().requires_grad_(True)
        k_fa = key.detach().clone().requires_grad_(True)
        v_fa = value.detach().clone().requires_grad_(True)
        fa_o, fa_t, fa_m = test("flash_attention", q_fa, k_fa, v_fa, prefix=True, prefix_lens=prefix_lens[:,:batch_size])
        assert torch.allclose(o, fa_o, rtol=1e-2, atol=1e-2)
        #print(fa_o)
        fa_o.backward(g)
        print("fa_o shape is ", fa_o.size())

        diff = fa_o - o
        #print(diff)
        print("q_py.grad is ",query.grad)
        print("q_fa.grad is ", q_fa.grad)
        print(f"custom pytorch time: {t:.6f}, peak memory: {m} MB")
        print(f"flash attention time: {fa_t:.6f}, speedup: {t/fa_t:.2f}; peak memory: {fa_m} MB, save: {int((m-fa_m)/m*100)}%")
        print("py-flash max is ", torch.max(o - fa_o))
        print("py-flash min is ", torch.min(o - fa_o))
        print("ref-flash max is ", torch.max(o_ref - fa_o))
        print("ref-flash min is ", torch.min(o_ref - fa_o))
        print("ref-py max is ", torch.max(o_ref - o))
        print("ref-py min is ", torch.min(o_ref - o))


        print(f'py-flash dQ max diff: {(q_py.grad - q_fa.grad).abs().max().item()}')
        print(f'py-flash dK max diff: {(k_py.grad - k_fa.grad).abs().max().item()}')
        print(f'py-flash dV max diff: {(v_py.grad - v_fa.grad).abs().max().item()}')
        print(f'ref-flash dQ max diff: {(query.grad - q_fa.grad).abs().max().item()}')
        print(f'ref-flash dK max diff: {(key.grad - k_fa.grad).abs().max().item()}')
        print(f'ref-flash dV max diff: {(value.grad - v_fa.grad).abs().max().item()}')

        print(f'ref-py dQ max diff: {(q_py.grad - query.grad).abs().max().item()}')
        print(f'ref-py dK max diff: {(k_py.grad - key.grad).abs().max().item()}')
        print(f'ref-py dV max diff: {(v_py.grad - value.grad).abs().max().item()}')