import jax
import torch
import jax.numpy as jnp
import numpy as np

from functools import partial
from timeit import timeit
from jax.experimental import pallas as pl
from flax import linen as nn

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide CUBLAS support error.

embd_dim = 768
head_dim = 64
heads = 12
exp_dim = 4

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)

A = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))
B = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))
C = jax.random.normal(key2, shape=(heads, embd_dim, head_dim))
O = jax.random.normal(key2, shape=(heads, head_dim, embd_dim))
W1 = jax.random.normal(key, shape=(heads, head_dim, head_dim * exp_dim))
W2 = jax.random.normal(key, shape=(heads, head_dim * exp_dim, head_dim))

def timing(f, *args):
    f_time = lambda: f(*args)
    timeit(f_time, number=20)
    return timeit(f_time, number=100)

def diff_gelu(x):
    tanh_out = jnp.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

chunk_dict = {
    'chunk_size': [],
    'results': [],
}

chunks = [16]
sequences = [16]

for chunk_size in chunks:

    res_dict = {
        'sequence_length': [],
        'dual': [],
        'dual_kernel': [],
    }

    for seq_len in sequences:

        sequence = jax.random.normal(key, shape=(heads, seq_len, embd_dim))

        @jax.jit
        def dual(sequence, A, B, C, O, W1, W2):
            
            @jax.vmap
            def parallelize_over_heads(X, XA, XB, XC, W1, W2):

                def body(carry, inputs):
                    W1, W2 = carry
                    X_chunk, XA_chunk, XB_chunk, XC_chunk = inputs

                    X1 = XB_chunk
                    Z = X1 @ W1
                    X2 = nn.gelu(Z)
                    Z2 = X2 @ W2
                    L = Z2 - XA_chunk

                    P2 = L
                    P1 = P2 @ W2.T * diff_gelu(Z)
                    
                    Attn1 = jnp.tril(XC_chunk @ XB_chunk.T)
                    Z1_bar = XC_chunk @ W1 - Attn1 @ P1
                    X2_bar = nn.gelu(Z1_bar)
                    
                    Attn2 = jnp.tril(X2_bar @ X2_bar.T)
                    Z2_bar = X2_bar @ W2 - Attn2 @ P2
                    XCW_chunk = Z2_bar

                    W1_new = W1 - X1.T @ P1
                    W2_new = W2 - X2.T @ P2
                    
                    return (W1_new, W2_new), XCW_chunk

                inputs = X, XA, XB, XC
                _, XCW = jax.lax.scan(body, (W1, W2), inputs)
                XCW = jnp.reshape(XCW, (seq_len, embd_dim))
                return XCW

            X = sequence.reshape((heads, -1, chunk_size, embd_dim))
            XA = sequence @ A
            XB = sequence @ B 
            XC = sequence @ C


            print("X Shape: ", X.shape)
            print("A Shape: ", A.shape)
            print("XA Shape: ", XA.shape)
            print("XB Shape: ", XB.shape)
            print("XC Shape: ", XC.shape)
            XCW = parallelize_over_heads(X, XA, XB, XC, W1, W2)
            return XCW @ O

test = dual(sequence, A, B, C, O, W1, W2)