import glob
import os
import re
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from model import FullHybridMoEModel, ModelConfig
from optimizer import compute_loss, make_hybrid_optimizer
from utils import path_to_str


def make_tpu_mesh():
    devices = jax.devices()
    n = len(devices)
    mesh_devices = mesh_utils.create_device_mesh((n,), devices)
    return Mesh(mesh_devices, axis_names=("tpu_nodes",))


def make_shard_and_compile(config: ModelConfig, total_steps: int, batch_size: int):
    mesh = make_tpu_mesh()
    n_devices = mesh.shape["tpu_nodes"]

    if batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by the number of devices "
            f"({n_devices}); data is sharded along the batch axis across devices."
        )

    tx = make_hybrid_optimizer(total_steps=total_steps)
    model = FullHybridMoEModel(cfg=config)

    init_rng = jax.random.PRNGKey(0)
    # ВАЖНО: вычисление абстрактных параметров должно происходить ВНУТРИ mesh контекста,
    # чтобы shard_map (используемый внутри MLAJ) мог получить текущий mesh.
    with jax.set_mesh(mesh):
        abstract_params = jax.eval_shape(
            lambda: model.init(init_rng, jnp.zeros((batch_size, 8192), dtype=jnp.int32))
        )["params"]

   
    data_sharding = NamedSharding(mesh, P("tpu_nodes", None))

    # Pure data parallelism: EVERY param (including MoE expert weights) is fully
    # replicated on every chip.
    #
    # Раньше `experts_block` веса шардировались по первой оси (по числу экспертов),
    # что подразумевает expert-parallelism: каждый чип держит только часть экспертов,
    # а токены с других чипов должны физически доехать до "своего" эксперта через
    # all-to-all communication. Этот all-to-all нигде не был реализован -- MoEJ
    # обрабатывал только локальный на данном чипе батч. В результате код требовал от
    # компилятора невозможного: посчитать все E=8 экспертов "как будто" каждый чип
    # видит их все, но при этом веса экспертов физически лежали лишь частично на
    # каждом чипе. Это одна из причин, по которой параллелизация на 8 TPU не давала
    # никакого выигрыша по памяти -- компилятор был вынужден реплицировать/собирать
    # данные, а не честно шардировать вычисление.
    #
    # Замена на полную репликацию весов экспертов -- самый простой рабочий вариант:
    # каждый чип независимо считает MoE-роутинг для своего локального шардированного
    # батча (data parallel), без какой-либо кросс-чиповой коммуникации токенов.
    # Настоящий expert-parallelism (все-to-all + шардирование весов экспертов) можно
    # добавить позже отдельным, более сложным шагом, если 8x репликация памяти
    # экспертов станет узким местом -- пока модель всего 1.76B, это не так.
    def _get_param_sharding(path, param):
        return NamedSharding(mesh, P(*([None] * param.ndim)))

    param_sharding = jax.tree_util.tree_map_with_path(_get_param_sharding, abstract_params)

    opt_state_abstract = jax.eval_shape(lambda: tx.init(abstract_params))
    opt_state_sharding = jax.tree_util.tree_map_with_path(_get_param_sharding, opt_state_abstract)

    
    def distributed_train_step(p, s, b, r):
        loss_fn = lambda param: compute_loss(
            param, model.apply, b, config, rngs={"dropout": r}, deterministic=False, return_aux=True
        )
        (loss, aux_info), grads = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, new_s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), new_s, loss, aux_info

    @jax.jit
    def distributed_val_step(p, b):
        return compute_loss(p, model.apply, b, config, rngs=None, deterministic=True)

    # aux_info is a DICT with MIXED ranks -- ce_loss/aux_loss/z_loss are scalars
    # (rank 0, P() is correct), but expert_utilization has shape (num_layers,
    # num_experts) -- rank 2. A single NamedSharding(mesh, P()) applied to the
    # whole aux_info pytree tries to use a rank-0 spec for that rank-2 leaf.
    aux_info_sharding = {
        "ce_loss": NamedSharding(mesh, P()),
        "aux_loss": NamedSharding(mesh, P()),
        "z_loss": NamedSharding(mesh, P()),
        "expert_utilization": NamedSharding(mesh, P(None, None)),
    }
    compiled_train = jax.jit(
        distributed_train_step,
        in_shardings=(
            param_sharding,
            opt_state_sharding,
            {"input_ids": data_sharding, "labels": data_sharding},
            NamedSharding(mesh, P(None)),
        ),
        out_shardings=(
            param_sharding,
            opt_state_sharding,
            NamedSharding(mesh, P()),
            aux_info_sharding,
        ),
    )
    compiled_val = jax.jit(
        distributed_val_step,
        in_shardings=(param_sharding, {"input_ids": data_sharding, "labels": data_sharding}),
        out_shardings=NamedSharding(mesh, P()),
    )
    return compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding


def resolve_source_files(output_dir, prefix):
    merged_ids = os.path.join(output_dir, f"{prefix}_input_ids.npy")
    merged_lbls = os.path.join(output_dir, f"{prefix}_labels.npy")
    if os.path.exists(merged_ids) and os.path.exists(merged_lbls):
        return [(merged_ids, merged_lbls)]

    shard_ids_paths = sorted(
        glob.glob(os.path.join(output_dir, f"{prefix}_shard_ids_*.npy")),
        key=lambda p: int(re.search(r"_(\d+)\.npy$", p).group(1)),
    )
    pairs = []
    for ids_path in shard_ids_paths:
        lbls_path = ids_path.replace("_shard_ids_", "_shard_lbls_")
        if os.path.exists(lbls_path):
            pairs.append((ids_path, lbls_path))
    if not pairs:
        raise FileNotFoundError(
            f"Не найдены файлы для prefix={prefix!r} в {output_dir} -- ни объединённого "
            f"{prefix}_input_ids.npy, ни шардов {prefix}_shard_ids_*.npy. Проверьте путь."
        )
    return pairs


def build_manifest(file_pairs):
    manifest = []
    total = 0
    for ids_path, lbls_path in file_pairs:
        n_rows = np.load(ids_path, mmap_mode="r").shape[0]
        manifest.append((ids_path, lbls_path, n_rows))
        total += n_rows
        print(f"[DATA] {os.path.basename(ids_path)}: {n_rows:,} блоков")
    print(f"[DATA] Комбинированный пул: {total:,} блоков из {len(manifest)} файл(ов)")
    return manifest


def dataloader_multi_source(file_pairs, batch_size, data_sharding, val_split=0.05):
    manifest = build_manifest(file_pairs)
    sizes = np.array([n for _, _, n in manifest])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    total_blocks = int(offsets[-1])
    context_length = np.load(manifest[0][0], mmap_mode="r").shape[1]

    mmap_cache = {}

    def _get_mmap(path):
        arr = mmap_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            mmap_cache[path] = arr
        return arr

    def _gather_batch(global_indices):
        shard_of = np.searchsorted(offsets, global_indices, side="right") - 1
        ids_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        lbls_out = np.empty((len(global_indices), context_length), dtype=np.int32)
        for s in np.unique(shard_of):
            m = shard_of == s
            local_idx = global_indices[m] - offsets[s]
            ids_path, lbls_path, _ = manifest[s]
            ids_out[m] = _get_mmap(ids_path)[local_idx]
            lbls_out[m] = _get_mmap(lbls_path)[local_idx]
        return ids_out, lbls_out

    val_size = int(total_blocks * val_split)
    train_size = total_blocks - val_size

    all_idx = np.arange(total_blocks)
    np.random.RandomState(42).shuffle(all_idx)
    train_idx_pool = all_idx[:train_size]
    val_idx_pool = all_idx[train_size:]

    def _generator(pool, is_train=True):
        idx_local = np.copy(pool)
        local_rng = np.random.RandomState(123)
        while True:
            if is_train:
                local_rng.shuffle(idx_local)
            for step in range(len(idx_local) // batch_size):
                batch_idx = idx_local[step * batch_size: (step + 1) * batch_size]
                ids_np, lbls_np = _gather_batch(batch_idx)
                yield {
                    "input_ids": jax.device_put(jnp.array(ids_np), data_sharding),
                    "labels": jax.device_put(jnp.array(lbls_np), data_sharding),
                }
            if not is_train:
                break

    return (
        _generator(train_idx_pool, True),
        lambda: _generator(val_idx_pool, False),
        train_size // batch_size,
        val_size // batch_size,
    )


def main_execution():
    config = ModelConfig()

    file_pairs = [
        (
            "/kaggle/input/datasets/akseleu1j/agentic-datasetids-and-labels/processed_jax_data/agentic_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/agentic-datasetids-and-labels/processed_jax_data/agentic_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/coding-ids/coding_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/coding-labels/coding_labels.npy",
        ),
        (
            "/kaggle/input/datasets/akseleu1j/reasoning-ids/reasoning_input_ids.npy",
            "/kaggle/input/datasets/akseleu1j/reasoning-labels/reasoning_labels.npy",
        ),
        (    "/kaggle/input/datasets/akseleu1j/math-ids/new_data_ids.npy",
            "/kaggle/input/datasets/akseleu1j/math-labels/new_data_labels.npy",
            
        )
    ]

    for ids_path, lbls_path in file_pairs:
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"Не найден файл: {ids_path}")
        if not os.path.exists(lbls_path):
            raise FileNotFoundError(f"Не найден файл: {lbls_path}")
    print("Все файлы найдены.")

    manifest = build_manifest(file_pairs)
    total_blocks = sum(n for _, _, n in manifest)
    total_tokens = total_blocks * 8192
    print(f"Всего блоков: {total_blocks:,} (~= {total_tokens / 1e9:.2f} млрд токенов)")

    batch_size = 8
    epochs = 4
    early_stop_patience = 2
    eval_every_steps = 1000
    eval_batches = 150
    eval_patience = 4

    train_steps_per_epoch = (int(total_blocks * 0.95)) // batch_size
    total_train_steps = train_steps_per_epoch * epochs

    print(f"[TPU] Компиляция XLA графа под {total_train_steps} общих шагов обучения...")
    compiled_train, compiled_val, mesh, tx, model, param_sharding, opt_state_sharding, data_sharding = (
        make_shard_and_compile(config, total_train_steps, batch_size)
    )
    print(f"[TPU] Устройств в mesh: {mesh.shape['tpu_nodes']} (данные шардируются по батчу).")

    train_stream, val_factory, train_steps, val_steps = dataloader_multi_source(
        file_pairs, batch_size, data_sharding
    )

    global_rng = jax.random.PRNGKey(42)
    with jax.set_mesh(mesh):
        init_params_fn = jax.jit(
            lambda rng: model.init(rng, jnp.zeros((batch_size, 8192), dtype=jnp.int32))["params"],
            out_shardings=param_sharding,
        )
        params = init_params_fn(global_rng)
        print(f"[MEM] Доступно памяти на чипе 0: {jax.local_devices()[0].memory_stats()}")
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Общее количество параметров: {total_params:,} (≈ {total_params / 1e9:.2f} млрд)")

        weights_bytes = sum(x.nbytes for x in jax.tree_util.tree_leaves(params))
        print(f"Размер весов модели (на чип, при полной репликации): {weights_bytes / 1e9:.2f} ГБ")

        opt_state = jax.jit(lambda p: tx.init(p), out_shardings=opt_state_sharding)(params)

    checkpoint_dir = "/kaggle/working/orbax_checkpoints"
    options = ocp.CheckpointManagerOptions(max_to_keep=3, create=True)
    mngr = ocp.CheckpointManager(checkpoint_dir, ocp.StandardCheckpointer(), options)
    best_checkpoint_dir = "/kaggle/working/orbax_checkpoints_best"
    best_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=True)
    best_mngr = ocp.CheckpointManager(best_checkpoint_dir, ocp.StandardCheckpointer(), best_options)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    best_eval_loss = float("inf")
    eval_no_improve_count = 0
    stopped_early = False

    total_tokens_processed = 0
    epoch_start_time = time.perf_counter()

    for epoch in range(epochs):
        with jax.set_mesh(mesh):
            for step in range(train_steps):
                global_rng, step_rng = jax.random.split(global_rng)
                batch = next(train_stream)
                total_tokens_processed += batch_size * 8192

                params, opt_state, train_loss, aux_info = compiled_train(params, opt_state, batch, step_rng)
                global_step += 1

                if step % 10 == 0:
                    print(
                        f"Epoch: {epoch} | Step: {step}/{train_steps} | "
                        f"Global Step: {global_step} | Train Loss: {jax.device_get(train_loss):.4f} "
                        f"(ce={jax.device_get(aux_info['ce_loss']):.4f} "
                        f"aux={jax.device_get(aux_info['aux_loss']):.4f} "
                        f"z={jax.device_get(aux_info['z_loss']):.5f})"
                    )
                    if aux_info["expert_utilization"] is not None:
                        util = jax.device_get(aux_info["expert_utilization"])
                        util_std_per_layer = util.std(axis=-1)
                        worst_layer = int(util_std_per_layer.argmax())
                        print(
                            f"           expert utilization std (max over layers, layer {worst_layer}): "
                            f"{util_std_per_layer[worst_layer]:.4f} | ideal ~= 0, uniform = 1/{config.num_experts}"
                        )

                if global_step % eval_every_steps == 0:
                    val_stream = val_factory()
                    eval_loss = 0.0
                    n_batches_done = 0
                    for _ in range(eval_batches):
                        try:
                            eval_batch = next(val_stream)
                        except StopIteration:
                            break
                        eval_loss += jax.device_get(compiled_val(params, eval_batch))
                        n_batches_done += 1
                    eval_loss /= max(n_batches_done, 1)
                    print(f"[EVAL] Step {global_step}: val loss (частичный, {n_batches_done} батчей) = {eval_loss:.4f}")

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        eval_no_improve_count = 0
                    else:
                        eval_no_improve_count += 1
                        if eval_no_improve_count >= eval_patience:
                            print(
                                f"[EARLY STOP] Частичный val loss не улучшался {eval_patience} "
                                "проверок подряд. Останавливаю обучение немедленно."
                            )
                            mngr.save(global_step, args=ocp.args.StandardSave(params))
                            best_mngr.save(global_step, args=ocp.args.StandardSave(params))
                            print(f"[ORBAX] Финальный чекпоинт (шаг {global_step}) сохранён в оба каталога.")
                            stopped_early = True
                            break

            if stopped_early:
                break

            print(f"--- Эпоха {epoch} завершена. Запуск распределенной кросс-валидации ---")
            val_stream = val_factory()
            total_val_loss = 0.0
            for _ in range(val_steps):
                total_val_loss += jax.device_get(compiled_val(params, next(val_stream)))

            mean_val_loss = total_val_loss / val_steps
            print(f"===> Эпоха: {epoch} | ИТОГОВЫЙ СРЕДНИЙ VALIDATION LOSS: {mean_val_loss:.4f} <===")

            epoch_elapsed = time.perf_counter() - epoch_start_time
            tokens_per_sec = total_tokens_processed / epoch_elapsed
            print(f"Средняя скорость эпохи: {tokens_per_sec / 1e6:.2f} млн токенов/сек")
            total_tokens_processed = 0
            epoch_start_time = time.perf_counter()

            mngr.save(global_step, args=ocp.args.StandardSave(params))
            print(f"[ORBAX] Чекпоинт для шага {global_step} успешно зафиксирован.")

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
                best_mngr.save(global_step, args=ocp.args.StandardSave(params))
                print(f"[ORBAX] Новый лучший val loss ({best_val_loss:.4f}) -- сохранён в {best_checkpoint_dir}")
            else:
                epochs_without_improvement += 1
                print(
                    f"[EARLY STOP] val loss не улучшился {epochs_without_improvement} эпох(и) подряд "
                    f"(лучший: {best_val_loss:.4f})"
                )
                if epochs_without_improvement >= early_stop_patience:
                    print(
                        f"[EARLY STOP] Останавливаю обучение -- val loss не улучшался "
                        f"{early_stop_patience} эпохи подряд. Лучшие веса лежат в {best_checkpoint_dir}."
                    )
                    break

    print("Обучение завершено.")


if __name__ == "__main__":
    main_execution()
