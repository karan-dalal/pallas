from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np

def mlp(x, w1, w2, y):
  hidden1 = jnp.dot(x, w1)
  hidden2 = jax.nn.relu(hidden1)
  output = jnp.dot(hidden2, w2)
  loss = jnp.mean(jnp.square(output - y))
  return loss

grad_func = jax.grad(mlp, argnums=(1))

def gradient_kernel(x_ref, w1_ref, w2_ref, y_ref, output_ref):
  x, w1, w2, y = x_ref[...], w1_ref[...], w2_ref[...], y_ref[...]
  grad_mlp = jax.grad(mlp, argnums=(1))
#   grad_w1 = grad_func(x, w1, w2, y)
  _, vjp_func = jax.vjp(mlp, x, w1, w2, y)
  _, grad_w1, _, _ = vjp_func(0.0)
  output_ref[...] = grad_w1

@jax.jit
def gradient_kernel_test(x: jax.Array, w1: jax.Array, w2: jax.Array, y: jax.Array) -> jax.Array:
  return pl.pallas_call(gradient_kernel,
                        out_shape=jax.ShapeDtypeStruct(w1.shape, w1.dtype)
                        )(x, w1, w2, y)

key = jax.random.PRNGKey(0)

x = jax.random.normal(key, shape=(5,))
w1 = jax.random.normal(key, shape=(5,5))
w2 = jax.random.normal(key, shape=(5,5))
y = jax.random.normal(key, shape=(5,))

# Use Pallas kernel or standard JAX.
use_kernel = True

if use_kernel:
  grad_w1 = gradient_kernel_test(x, w1, w2, y)
else:
  grad_mlp = jax.grad(mlp, argnums=(1))
  grad_w1 = grad_mlp(x, w1, w2, y)

print(grad_w1)