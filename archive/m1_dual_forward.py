import jax
# import torch
import jax.numpy as jnp
import numpy as np
import pdb

from functools import partial
from timeit import timeit
from jax.experimental import pallas as pl

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide CUBLAS support error.

seq_len, chunk_size = 16, 16
embd_dim = 2048
head_dim = 128 
heads = 16

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)

sequence = jax.random.normal(key, shape=(seq_len, embd_dim))
kernel_output = jnp.zeros([seq_len, head_dim])

A = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))
B = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))
C = jax.random.normal(key2, shape=(heads, embd_dim, head_dim))
O = jax.random.normal(key2, shape=(heads, head_dim, embd_dim))

W0 = jax.random.normal(key, shape=(heads, head_dim, head_dim))
b0 = jax.random.normal(key, shape=(heads, head_dim))

ln_scale = jax.random.normal(key, shape=(heads, head_dim))
ln_bias = jax.random.normal(key, shape=(heads, head_dim))

def timing(f, *args):
    f_time = lambda: f(*args)
    timeit(f_time, number=20)
    return timeit(f_time, number=100)

def layer_norm(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + 1e-6) + bias

@jax.jit
def dual(sequence, A, B, C, O, W0, b0, ln_scale, ln_bias):
    sequenceA = sequence @ A
    sequenceB = sequence @ B
    sequenceC = sequence @ C

    sequenceChunked_A = jnp.reshape(sequenceA, (heads, seq_len // chunk_size, chunk_size, head_dim))
    sequenceChunked_B = jnp.reshape(sequenceB, (heads, seq_len // chunk_size, chunk_size, head_dim))
    sequenceChunked_C = jnp.reshape(sequenceC, (heads, seq_len // chunk_size, chunk_size, head_dim))

    @jax.vmap
    def parallelize_over_heads(headChunked_A, headChunked_B, headChunked_C, W0, b0, head_scale, head_bias):
        
        def compute_chunk(carry, inputs):
            XA_chunk, XB_chunk, XC_chunk = inputs # [16, 128]
            W1_init, b1_init = carry

            X1 = XB_chunk
            Z1 = X1 @ W1_init + b1_init

            DLN_out, LN_vjp = jax.vjp(lambda z: layer_norm(z, head_scale, head_bias), Z1)
            grad_l_wrt_DLN_out = DLN_out - XA_chunk  # [K,f]
            grad_l_wrt_Z1 = LN_vjp(grad_l_wrt_DLN_out)[0]  # [K,f]

            Attn1 = jnp.tril(XC_chunk @ XB_chunk.transpose(1, 0))
            b1_bar = b1_init -  jnp.cumsum(grad_l_wrt_Z1, axis=0)  # [K,f]
            Z1_bar = XC_chunk @ W1_init - (Attn1) @ grad_l_wrt_Z1 + b1_bar
            Z1_bar = layer_norm(Z1_bar, head_scale, head_bias)

            XCW_chunk = Z1_bar

            W1_last = W1_init - (X1).transpose(1, 0) @ grad_l_wrt_Z1
            b1_last = b1_bar[-1]

            return (W1_last, b1_last), XCW_chunk

        W_final, XCW = jax.lax.scan(compute_chunk, (W0, b0), (headChunked_A, headChunked_B, headChunked_C))
        return XCW
    
    sequenceCW = parallelize_over_heads(sequenceChunked_A, sequenceChunked_B, sequenceChunked_C, W0, b0, ln_scale, ln_bias)
    output = jnp.reshape(sequenceCW, (heads, seq_len, head_dim)) @ O
    return output

def m1_forward(XA_ref, XB_ref, XC_ref, W0_ref, b0_ref, scale_ref, bias_ref, o_ref):
    seq_len, head_dim = XA_ref.shape
    W0, b0 = W0_ref[:], b0_ref[:]
    head_scale, head_bias = scale_ref[:], bias_ref[:]

    CHUNK_SIZE = 16

    def compute_chunk(i, carry):
        XA_chunk = pl.load(XA_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)))
        XB_chunk = pl.load(XB_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)))
        XC_chunk = pl.load(XC_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)))

        W1_init, b1_init = carry

        X1 = XB_chunk
        Z1 = X1 @ W1_init + b1_init

        DLN_out, LN_vjp = jax.vjp(lambda z: layer_norm(z, head_scale, head_bias), Z1)
        grad_l_wrt_DLN_out = DLN_out - XA_chunk 
        grad_l_wrt_Z1 = LN_vjp(grad_l_wrt_DLN_out)[0]

        Attn1 = jnp.tril(XC_chunk @ XB_chunk.transpose(1, 0))
        b1_bar = b1_init -  jnp.cumsum(grad_l_wrt_Z1, axis=0)
        Z1_bar = XC_chunk @ W1_init - (Attn1) @ grad_l_wrt_Z1 + b1_bar
        Z1_bar = layer_norm(Z1_bar, head_scale, head_bias)

        XCW_chunk = Z1_bar
        pl.store(o_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)), XCW_chunk)

        W1_last = W1_init - (X1).transpose(1, 0) @ grad_l_wrt_Z1
        b1_last = b1_init -  jnp.sum(grad_l_wrt_Z1, axis=0)

        return (W1_last, b1_last)
    
    W = jax.lax.fori_loop(0, seq_len // CHUNK_SIZE, compute_chunk, (W0, b0))

launch_kernel = pl.pallas_call(m1_forward, out_shape=jax.ShapeDtypeStruct(kernel_output.shape, kernel_output.dtype))

@jax.jit
def kernel(sequence, A, B, C, O, W0, b0, ln_scale, ln_bias):
    sequenceA = sequence @ A
    sequenceB = sequence @ B
    sequenceC = sequence @ C

    @jax.vmap
    def parallelize_over_heads(headChunked_A, headChunked_B, headChunked_C, W0, b0, head_scale, head_bias):
        return launch_kernel(headChunked_A, headChunked_B, headChunked_C, W0, b0, head_scale, head_bias)
    
    sequenceCW = parallelize_over_heads(sequenceA, sequenceB, sequenceC, W0, b0, ln_scale, ln_bias)
    output = jnp.reshape(sequenceCW, (heads, seq_len, head_dim)) @ O
    return output

kernel(sequence, A, B, C, O, W0, b0, ln_scale, ln_bias)
