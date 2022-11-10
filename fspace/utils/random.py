import jax


def tree_split(key, tree):
    treedef = jax.tree_util.tree_structure(tree)
    key, *key_list = jax.random.split(key, 1 + treedef.num_leaves)
    return key, jax.tree_util.tree_unflatten(treedef, key_list)
