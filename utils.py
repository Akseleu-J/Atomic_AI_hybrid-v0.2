def path_to_str(path) -> str:
    """Turn a jax.tree_util key-path into a lowercase string for substring matching.

    `jax.tree_util.tree_map_with_path` hands back a tuple of key objects whose type
    depends on the container at that point in the pytree:
      - DictKey            (dict / FrozenDict entries)   -> has `.key`
      - FlattenedIndexKey   (flattened containers)         -> has `.key`
      - GetAttrKey          (NamedTuple fields)             -> has `.name`
      - SequenceKey         (list / plain tuple entries)   -> has `.idx`

    A flax params pytree is dict-only, so `str(p.key)` alone happens to work there.
    An optax optimizer-state pytree (from `optax.chain`/`multi_transform`/NamedTuple
    states like `MuonState`) mixes in GetAttrKey and SequenceKey entries, and `p.key`
    raises `AttributeError: 'SequenceKey' object has no attribute 'key'` on those.
    This checks all the attribute names generically instead of assuming `.key`.
    """
    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "".join(parts).lower()


def collect_by_leaf_name(tree, target_name):
    """Collect every leaf in a pytree whose final key-path segment equals target_name.

    flax's `self.sow(col, name, value)` stores `value` nested under the calling
    module's own scope path -- the sown collections mirror the FULL module
    hierarchy, exactly like the `params` collection does (confirmed in Flax's own
    docs: "the different variable collections share the same nested tree structure").
    So a value sown as `self.sow("losses", "aux_loss", v)` inside `layer_5/moe` ends
    up at `variables["losses"]["layer_5"]["moe"]["aux_loss"]` -- NOT at a flat
    top-level `variables["losses"]["aux_loss"]`. When the same name is sown once per
    layer (e.g. one MoE call per transformer layer), this walks the whole nested
    dict and pulls out every layer's value regardless of nesting depth, so callers
    don't have to know or hard-code the exact module-path structure.
    """
    import jax  # local import: this module has no other jax dependency

    collected = []

    def _mark(path, leaf):
        if path and path_to_str([path[-1]]) == target_name.lower():
            collected.append(leaf)
        return leaf

    jax.tree_util.tree_map_with_path(_mark, tree)
    return collected
