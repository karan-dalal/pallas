import jax
import torch
import jax.numpy as jnp
import numpy as np

from functools import partial
from timeit import timeit
from jax.experimental import pallas as pl

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide CUBLAS support error.

embd_dim = 768
head_dim = 64
heads = 12

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)

A = jax.random.normal(key1, shape=(heads, head_dim, embd_dim))
B = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))
C = jax.random.normal(key2, shape=(heads, head_dim, embd_dim))
D = jax.random.normal(key2, shape=(heads, embd_dim, head_dim))
W0 = jax.random.normal(key, shape=(heads, head_dim, head_dim))

def timing(f, *args):
    f_time = lambda: f(*args)
    timeit(f_time, number=20)
    return timeit(f_time, number=100)

chunk_dict = {
    'chunk_size': [],
    'results': [],
}

# chunks = [2**i for i in range(2, 10, 1)]
chunks = [1]
sequences = [1024]


for chunk_size in chunks:

    res_dict = {
        'sequence_length': [],
        'scan': [],
        'cumsum': [],
        'kernel_hadamard': [],
        'kernel_16': [],
    }

    for seq_len in sequences:

        sequence = jax.random.normal(key, shape=(heads, seq_len, embd_dim))
        kernel_output = jnp.zeros([chunk_size + head_dim, head_dim])

        @jax.jit
        @jax.vmap
        def scan_forward(sequence, A, B, C, D, W0):
            
            def inner_gradient(W, token):
                token_transformed = token @ B @ W @ A
                return 0.5 * ((token_transformed - token) ** 2).mean()

            inner_grad = jax.grad(inner_gradient, argnums=0)

            def body(carry, token_chunk):
                W = carry
                    
                grad_chunk = jax.vmap(partial(inner_grad, W))(token_chunk)
                token_chunk = token_chunk @ D

                def inner_forward(carry, input):
                    W = carry
                    token, gradient = input
                    
                    W_new = W - gradient
                    output = token @ W_new

                    return W_new, output

                W_final, token_transformed = jax.lax.scan(inner_forward, W, (token_chunk, grad_chunk))
                token_transformed = token_transformed @ C
                
                return W_final, token_transformed

            sequence = jnp.reshape(sequence, (seq_len // chunk_size, chunk_size, embd_dim))
            W_final, output = jax.lax.scan(body, W0, sequence)
            output = jnp.reshape(output, (seq_len, embd_dim))
            return output

        @jax.jit
        @jax.vmap
        def cumsum_forward(sequence, A, B, C, D, W0):
            
            def inner_gradient(W, token):
                token_transformed = token @ B @ W @ A
                return 0.5 * ((token_transformed - token) ** 2).mean()

            def inner_forward(W, token):
                return token @ D @ W @ C

            inner_grad = jax.grad(inner_gradient, argnums=0)

            def body(carry, token_chunk):
                W = carry
                    
                grad_chunk = jax.vmap(partial(inner_grad, W))(token_chunk)
                grad_cumsum = jnp.cumsum(grad_chunk, axis=0)
                W_new = W - grad_cumsum
                token_transformed = jax.vmap(inner_forward)(W_new, token_chunk)
                
                return W_new[-1], token_transformed

            sequence = jnp.reshape(sequence, (seq_len // chunk_size, chunk_size, embd_dim))
            W_final, output = jax.lax.scan(body, W0, sequence)
            output = jnp.reshape(output, (seq_len, embd_dim))
            return output

        # NOTE: Kernel using hadamard product.
        def kernel(token_ref, W_ref, grad_ref, o_ref):
            W = W_ref[:]
            chunk_size, head_dim = token_ref.shape

            def body(i, W_curr):
                token = pl.load(token_ref, (pl.dslice(i, 1), slice(None)))
                gradient = pl.load(grad_ref, (i, slice(None), slice(None)))

                W_new = W_curr - gradient

                output = jnp.sum(token.T * W_new, axis=0, keepdims=True)

                pl.store(o_ref, (pl.dslice(i, 1), slice(None)), output)
                return W_new

            W_chunk_final = jax.lax.fori_loop(0, chunk_size, body, W)
            pl.store(o_ref, (pl.dslice(chunk_size, head_dim), slice(None)), W_chunk_final)

        launch_kernel = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(kernel_output.shape, kernel_output.dtype))

        @jax.jit
        @jax.vmap
        def fused_forward(sequence, A, B, C, D, W0):
            
            def inner_gradient(W, token):
                token_transformed = token @ B @ W @ A
                return 0.5 * ((token_transformed - token) ** 2).mean()

            inner_grad = jax.grad(inner_gradient, argnums=0)

            def body(carry, token_chunk):
                W = carry
                
                grad_chunk = jax.vmap(partial(inner_grad, W))(token_chunk)
                token_chunk = token_chunk @ D

                output = launch_kernel(token_chunk, W, grad_chunk)
                token_transformed = output[:chunk_size] @ C
                W_new = output[chunk_size:]

                return W_new, token_transformed 

            sequence = jnp.reshape(sequence, (seq_len // chunk_size, chunk_size, embd_dim))
            W_final, output = jax.lax.scan(body, W0, sequence)
            output = jnp.reshape(output, (seq_len, embd_dim))
            return output

        # NOTE: Kernel using casting trick for tensor cores.
        def kernel_16(token_ref, W_ref, grad_ref, o_ref):
            W = W_ref[:]
            chunk_size, head_dim = token_ref.shape

            def body(i, W_curr):
                token = pl.load(token_ref, (pl.dslice(i, 1), slice(None)))
                gradient = pl.load(grad_ref, (i, slice(None), slice(None)))

                W_new = W_curr - gradient

                output = jnp.mean(token.repeat(16, axis=0) @ W_new, axis=0, keepdims=True)

                pl.store(o_ref, (pl.dslice(i, 1), slice(None)), output)
                return W_new

            W_chunk_final = jax.lax.fori_loop(0, chunk_size, body, W)
            pl.store(o_ref, (pl.dslice(chunk_size, head_dim), slice(None)), W_chunk_final)

        launch_kernel_16 = pl.pallas_call(kernel_16, out_shape=jax.ShapeDtypeStruct(kernel_output.shape, kernel_output.dtype))

        @jax.jit
        @jax.vmap
        def fused_forward_16(sequence, A, B, C, D, W0):
            
            def inner_gradient(W, token):
                token_transformed = token @ B @ W @ A
                return 0.5 * ((token_transformed - token) ** 2).mean()

            inner_grad = jax.grad(inner_gradient, argnums=0)

            def body(carry, token_chunk):
                W = carry
                
                grad_chunk = jax.vmap(partial(inner_grad, W))(token_chunk)
                token_chunk = token_chunk @ D

                output = launch_kernel_16(token_chunk, W, grad_chunk)
                token_transformed = output[:chunk_size] @ C
                W_new = output[chunk_size:]

                return W_new, token_transformed 

            sequence = jnp.reshape(sequence, (seq_len // chunk_size, chunk_size, embd_dim))
            W_final, output = jax.lax.scan(body, W0, sequence)
            output = jnp.reshape(output, (seq_len, embd_dim))
            return output

        res_dict['sequence_length'].append(seq_len)
        res_dict['scan'].append(timing(scan_forward, sequence, A, B, C, D, W0) * 1000)
        res_dict['cumsum'].append(timing(cumsum_forward, sequence, A, B, C, D, W0) * 1000)
        res_dict['kernel_hadamard'].append(timing(fused_forward, sequence, A, B, C, D, W0) * 1000)
        res_dict['kernel_16'].append(timing(fused_forward_16, sequence, A, B, C, D, W0) * 1000)

    chunk_dict['chunk_size'].append(chunk_size)
    chunk_dict['results'].append(res_dict)

print(chunk_dict)
# torch.save(chunk_dict, '/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_cumsum_forward_verified_chunk.pth') 