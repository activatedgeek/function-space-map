import jax


@jax.jit
def tree_split(key, ref_tree):
    treedef = jax.tree_util.tree_structure(ref_tree)
    key, *key_list = jax.random.split(key, 1 + treedef.num_leaves)
    return key, jax.tree_util.tree_unflatten(treedef, key_list)


@jax.jit
def sample_tree(key, ref_tree, mean, std):
    key, key_tree = tree_split(key, ref_tree)
    def _sample_param(key, param):
        return mean + std * jax.random.normal(key, param.shape, param.dtype)
    return key, jax.tree_util.tree_map(_sample_param, key_tree, ref_tree)
