"""
ModelRunnerProcess -- GPU-owning component that executes model forward passes.

Instantiated **in-process** by :class:`SchedulerProcess`
The scheduler calls :meth:`_forward_batch` directly —
no inter-process communication is involved.

This component owns the GPU: it holds a :class:`ModelRunner` with model
weights, KV-cache memory pools, and the attention backend.  It also owns
the :class:`RadixCache` for prefix-aware KV reuse.

RadixCache lifecycle
--------------------
1. **match_prefix** — called during ``_allocate_extend`` before KV allocation.
2. **inc_lock_ref** — locks matched radix-tree nodes to prevent eviction.
3. **insert (prefill)** — inserts prompt KV indices after prefill.
4. **insert (completion)** — re-inserts the full sequence when a request finishes.
5. **dec_lock_ref** — unlocks radix-tree nodes when a request is freed.
6. **evict** — called when KV allocation fails to free stale cache entries.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

from pymllm.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

logger = logging.getLogger(__name__)

# Fraction of KV pool to try evicting when allocation fails.
_EVICT_FRACTION = 0.10
# Maximum number of eviction retries before giving up.
_MAX_EVICT_RETRIES = 3


class ModelRunnerProcess:
    """GPU-owning component created in-process by SchedulerProcess."""

    def __init__(
        self,
        gpu_id: int = 0,
        server_config: Optional[Any] = None,
        model_config: Optional[Any] = None,
    ):
        self._gpu_id = gpu_id
        self._server_config = server_config
        self._model_config = model_config

        # The ModelRunner instance (created in init_model)
        self._runner = None
        self._is_hybrid: bool = False

        # RadixCache instance (created in init_model, after memory pools)
        self._radix_cache: Optional[RadixCache] = None

        # GPU resource tracking: maps rid -> req_pool_idx (slot in ReqToTokenPool)
        self._rid_to_req_pool_idx: Dict[str, int] = {}
        # Maps rid -> kv_indices tensor (all KV-cache token indices for this request)
        self._rid_to_kv_indices: Dict[str, torch.Tensor] = {}
        # Maps rid -> input_ids used for prefill (needed for radix cache insert)
        self._rid_to_input_ids: Dict[str, List[int]] = {}
        # Maps rid -> list of generated (decode) token ids, appended each step.
        # Used to build the full sequence for radix cache insert at completion.
        self._rid_to_output_ids: Dict[str, List[int]] = {}
        # Maps rid -> cache_protected_len: the length of the prefix that has
        # already been inserted into the radix cache.  When insert() returns
        # prefix_len > cache_protected_len, the KV indices in the overlap
        # range [cache_protected_len, prefix_len) are duplicates that must
        # be freed from the allocator (the tree already holds cloned copies).
        self._rid_to_cache_protected_len: Dict[str, int] = {}
        # Maps rid -> (last_node, swa_boundary_id) for radix cache lock tracking
        self._rid_to_radix_lock: Dict[str, Tuple[TreeNode, Optional[int]]] = {}
        # Maps rid -> mrope_position_delta (M-RoPE positional offset per request)
        # Populated during prefill; used to offset decode-step positions for
        # multimodal models (Qwen3-VL) that consume more position indices than
        # tokens due to 3-D image grid positions.
        self._rid_to_mrope_delta: Dict[str, int] = {}

        # GDN prefix cache state tracking (hybrid models only):
        # Maps rid -> GDN track slot index in GDNPool (for snapshotting state)
        self._rid_to_gdn_track_slot: Dict[str, int] = {}
        # Maps radix tree node id -> GDN track slot index
        self._node_id_to_gdn_track_slot: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init_model(self) -> None:
        """Create and initialise the ModelRunner and RadixCache.

        Must run inside the subprocess (after spawn) since it does CUDA init.
        """
        from pymllm.executor.model_runner import ModelRunner

        logger.info(
            "ModelRunnerProcess: initialising ModelRunner on GPU %d",
            self._gpu_id,
        )
        self._runner = ModelRunner(
            server_config=self._server_config,
            model_config=self._model_config,
            gpu_id=self._gpu_id,
        )
        self._runner.initialize()

        # Initialise RadixCache after memory pools are ready.
        disable_cache = getattr(self._server_config, "disable_radix_cache", False)
        self._is_hybrid = self._runner.num_gdn_layers > 0
        if self._is_hybrid and not disable_cache:
            logger.info(
                "ModelRunnerProcess: prefix caching ENABLED with GDN state "
                "tracking (%d GDN layers)",
                self._runner.num_gdn_layers,
            )
        sliding_window = self._runner.sliding_window_size
        page_size = getattr(self._server_config, "radix_cache_page_size", 1)
        # For hybrid models, register an eviction callback so that evicted
        # radix nodes free their associated GDN track slots.
        evict_cb = self._on_radix_node_evict if self._is_hybrid else None
        self._radix_cache = RadixCache(
            page_size=page_size,
            sliding_window_size=sliding_window,
            disable=disable_cache,
            token_to_kv_pool_allocator=self._runner.token_to_kv_pool_allocator,
            on_node_evict=evict_cb,
        )
        logger.info(
            "ModelRunnerProcess: RadixCache initialized "
            "(disable=%s, sliding_window=%s)",
            disable_cache,
            sliding_window,
        )
        logger.info("ModelRunnerProcess: ModelRunner ready")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Run the model forward pass and sampling for *batch*.

        *batch* is a dict produced by ``ScheduleBatch.to_batch_dict()``
        containing ``"forward_mode"``, ``"input_ids"``, ``"seq_lens"``,
        ``"req_pool_indices"``, ``"requests"`` (metadata list), etc.

        Implements 6 phases:
        1. Cleanup: free GPU resources for rids no longer in the batch
        2. Prefix matching + KV allocation
        3. Build GPU tensors
        4. Forward + sample
        5. Radix cache insert (extend only)
        6. Build result dict
        """
        runner = self._runner
        forward_mode = batch.get("forward_mode", "decode")
        batch_size = batch.get("batch_size", 0)
        requests_meta: List[Dict[str, Any]] = batch.get("requests", [])

        if batch_size == 0:
            return {"batch_id": batch.get("batch_id"), "outputs": []}

        device = runner.device

        # Collect current batch rids
        current_rids: Set[str] = {m["rid"] for m in requests_meta}

        # ==============================================================
        # Phase 2: Prefix matching + KV allocation
        # ==============================================================
        # For extend batches, match_prefix is done inside _allocate_extend
        # which may update extend_prefix_lens and extend_seq_lens.
        if forward_mode == "extend":
            out_cache_loc, actual_prefix_lens, actual_extend_lens = (
                self._allocate_extend(batch, requests_meta)
            )
        else:
            out_cache_loc = self._allocate_decode(batch, requests_meta)
            actual_prefix_lens = None
            actual_extend_lens = None

        # ==============================================================
        # Phase 3: Build GPU tensors
        # ==============================================================
        if forward_mode == "extend" and actual_prefix_lens is not None:
            # Rebuild input_ids and seq_lens using actual prefix matches.
            # The scheduler sent tokens assuming prefix_len=0; we need to
            # trim the input_ids to skip the prefix-matched tokens.
            (
                input_ids_tensor,
                seq_lens_tensor,
                extend_seq_lens_t,
                extend_prefix_lens_t,
            ) = self._rebuild_extend_tensors(
                batch, requests_meta, actual_prefix_lens, actual_extend_lens, device
            )
        else:
            input_ids_list: List[int] = batch["input_ids"]
            seq_lens_list: List[int] = batch["seq_lens"]
            input_ids_tensor = torch.tensor(
                input_ids_list, dtype=torch.int32, device=device
            )
            seq_lens_tensor = torch.tensor(
                seq_lens_list, dtype=torch.int32, device=device
            )
            extend_seq_lens_t = None
            extend_prefix_lens_t = None

        # Build req_pool_indices from our own tracking (NOT from scheduler)
        req_pool_indices = torch.tensor(
            [self._rid_to_req_pool_idx[m["rid"]] for m in requests_meta],
            dtype=torch.int64,
            device=device,
        )

        out_cache_loc = out_cache_loc.to(torch.int64)

        # ==============================================================
        # Phase 4: Forward + sample
        # ==============================================================
        # Extract per-request sampling params
        temperatures = []
        top_ps = []
        top_ks = []
        for m in requests_meta:
            sp = m.get("sampling_params") or {}
            temperatures.append(sp.get("temperature", 1.0))
            top_ps.append(sp.get("top_p", 1.0))
            top_ks.append(sp.get("top_k", -1))

        temps_tensor = torch.tensor(temperatures, dtype=torch.float32, device=device)
        top_ps_tensor = torch.tensor(top_ps, dtype=torch.float32, device=device)
        top_ks_tensor = torch.tensor(top_ks, dtype=torch.int32, device=device)

        if forward_mode == "extend":
            if extend_seq_lens_t is None:
                extend_seq_lens_list: List[int] = batch["extend_seq_lens"]
                extend_prefix_lens_list: List[int] = batch["extend_prefix_lens"]
                extend_seq_lens_t = torch.tensor(
                    extend_seq_lens_list, dtype=torch.int32, device=device
                )
                extend_prefix_lens_t = torch.tensor(
                    extend_prefix_lens_list, dtype=torch.int32, device=device
                )

            fb = runner.prepare_forward_batch_extend(
                input_ids=input_ids_tensor,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_tensor,
                extend_seq_lens=extend_seq_lens_t,
                extend_prefix_lens=extend_prefix_lens_t,
                out_cache_loc=out_cache_loc,
            )

            # Attach multimodal vision inputs to ForwardBatch so the
            # model's vision encoder can process images during prefill.
            # The tokenizer wraps processor output under "image_inputs";
            # fall back to top-level keys for direct dicts.
            pixel_values_list = []
            image_grid_thw_list = []
            for m in requests_meta:
                mm = m.get("mm_inputs")
                if mm is None:
                    continue
                # AutoProcessor output is nested under "image_inputs"
                src = mm.get("image_inputs") if "image_inputs" in mm else mm
                if src is None:
                    continue
                pv = (
                    src.get("pixel_values")
                    if hasattr(src, "get")
                    else getattr(src, "pixel_values", None)
                )
                thw = (
                    src.get("image_grid_thw")
                    if hasattr(src, "get")
                    else getattr(src, "image_grid_thw", None)
                )
                if pv is not None:
                    if not isinstance(pv, torch.Tensor):
                        pv = torch.as_tensor(pv)
                    pixel_values_list.append(pv.to(device=device))
                if thw is not None:
                    if not isinstance(thw, torch.Tensor):
                        thw = torch.as_tensor(thw)
                    image_grid_thw_list.append(thw.to(device=device))
            if pixel_values_list:
                fb.pixel_values = torch.cat(pixel_values_list, dim=0)
            if image_grid_thw_list:
                fb.image_grid_thw = torch.cat(image_grid_thw_list, dim=0)
        else:
            # Build mrope_position_deltas tensor for decode batches.
            mrope_deltas = [
                self._rid_to_mrope_delta.get(m["rid"], 0) for m in requests_meta
            ]
            mrope_deltas_tensor = torch.tensor(
                mrope_deltas, dtype=torch.int64, device=device
            )

            fb = runner.prepare_forward_batch_decode(
                input_ids=input_ids_tensor,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens_tensor,
                out_cache_loc=out_cache_loc,
                mrope_position_deltas=mrope_deltas_tensor,
            )

        logits_output = runner.forward(fb)

        # Persist M-RoPE position deltas for multimodal models (Qwen3-VL).
        # The model sets mrope_position_deltas on the ForwardBatch during
        # prefill; we store them here so decode steps can retrieve them.
        if (
            forward_mode == "extend"
            and getattr(fb, "mrope_position_deltas", None) is not None
        ):
            deltas_cpu = fb.mrope_position_deltas.cpu().tolist()
            for idx, m in enumerate(requests_meta):
                self._rid_to_mrope_delta[m["rid"]] = int(deltas_cpu[idx])

        next_token_ids = runner.sample(
            logits_output,
            fb,
            temperatures=temps_tensor,
            top_ps=top_ps_tensor,
            top_ks=top_ks_tensor,
        )

        # ==============================================================
        # Phase 4.5: Snapshot GDN state after extend (hybrid models)
        # ==============================================================
        if forward_mode == "extend" and self._is_hybrid:
            self._track_gdn_state_after_extend(requests_meta)

        # ==============================================================
        # Phase 5: Radix cache insert (extend only)
        # ==============================================================
        if forward_mode == "extend" and self._radix_cache is not None:
            self._insert_into_radix_cache(requests_meta)

        # ==============================================================
        # Phase 6: Build result & track output tokens
        # ==============================================================
        next_ids_cpu = next_token_ids.cpu().tolist()
        outputs: List[Dict[str, Any]] = []
        for i, m in enumerate(requests_meta):
            rid = m["rid"]
            token_id = next_ids_cpu[i] if i < len(next_ids_cpu) else 0
            # Track output tokens for radix cache insert at completion
            out_ids = self._rid_to_output_ids.get(rid)
            if out_ids is not None:
                out_ids.append(token_id)

            out: Dict[str, Any] = {
                "rid": rid,
                "output_token_ids": [token_id],
            }
            # Report actual prefix_len back to the scheduler so it can
            # update its token budget tracking accurately.
            if actual_prefix_lens is not None:
                out["prefix_len"] = actual_prefix_lens[i]
            outputs.append(out)

        return {
            "batch_id": batch.get("batch_id"),
            "outputs": outputs,
        }

    # ------------------------------------------------------------------
    # Tensor rebuild for prefix-matched extend
    # ------------------------------------------------------------------

    def _rebuild_extend_tensors(
        self,
        batch: Dict[str, Any],
        requests_meta: List[Dict[str, Any]],
        actual_prefix_lens: List[int],
        actual_extend_lens: List[int],
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rebuild input_ids and related tensors after prefix matching.

        The scheduler sent input_ids assuming no prefix cache hit.  After
        radix cache matching, we know the actual prefix lengths and must
        trim the input_ids accordingly.

        Returns (input_ids, seq_lens, extend_seq_lens, extend_prefix_lens)
        as GPU tensors.
        """
        # Reconstruct trimmed input_ids: for each request, take only the
        # tokens beyond the matched prefix.
        new_input_ids: List[int] = []
        seq_lens_list: List[int] = batch["seq_lens"]

        for i, m in enumerate(requests_meta):
            full_input_ids = m.get("input_ids", [])
            prefix_len = actual_prefix_lens[i]
            # Only send tokens after the prefix
            new_input_ids.extend(full_input_ids[prefix_len:])

        input_ids = torch.tensor(new_input_ids, dtype=torch.int32, device=device)
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        extend_seq_lens = torch.tensor(
            actual_extend_lens, dtype=torch.int32, device=device
        )
        extend_prefix_lens = torch.tensor(
            actual_prefix_lens, dtype=torch.int32, device=device
        )
        return input_ids, seq_lens, extend_seq_lens, extend_prefix_lens

    # ------------------------------------------------------------------
    # Radix cache insert
    # ------------------------------------------------------------------

    def _insert_into_radix_cache(self, requests_meta: List[Dict[str, Any]]) -> None:
        """Insert prefill KV indices into the radix cache for future reuse.

        1. **Insert** the request's token → KV index mapping into the tree.
        2. **Free duplicates** — indices in ``[cache_protected_len, new_prefix_len)``
           are now owned by the tree; the request's copies are redundant.
        3. **Re-match + write-back** — fetch the tree's *own* indices via
           ``match_prefix`` and write them into ``req_to_token_pool``,
           replacing the just-freed entries.  Without this step the pool
           still points at freed slots → use-after-free during decode.
        4. **Update** ``cache_protected_len`` and radix lock.
        """
        _dbg = logger.isEnabledFor(logging.DEBUG)
        cache = self._radix_cache
        if cache is None or cache.disable:
            return

        runner = self._runner
        gdn_pool = getattr(runner, "gdn_pool", None)

        for m in requests_meta:
            rid = m["rid"]
            input_ids = self._rid_to_input_ids.get(rid)
            if input_ids is None:
                continue

            slot = self._rid_to_req_pool_idx.get(rid)
            if slot is None:
                continue

            seq_len = len(input_ids)
            kv_indices = runner.req_to_token_pool.req_to_token[slot, :seq_len].to(
                torch.int64
            )

            if _dbg:
                logger.debug(
                    "[CACHE INSERT] rid=%s seq_len=%d pool[slot=%d,0:%d]=%s",
                    rid,
                    seq_len,
                    slot,
                    min(seq_len, 8),
                    kv_indices[: min(seq_len, 8)].tolist(),
                )

            key = RadixKey(input_ids)
            result = cache.insert(key, kv_indices)
            new_prefix_len = result.prefix_len

            # --- Step 2: free duplicates ---
            cache_protected_len = self._rid_to_cache_protected_len.get(rid, 0)
            if _dbg:
                logger.debug(
                    "[CACHE INSERT] rid=%s insert prefix_len=%d cache_protected=%d",
                    rid,
                    new_prefix_len,
                    cache_protected_len,
                )
            if new_prefix_len > cache_protected_len:
                dup_indices = kv_indices[cache_protected_len:new_prefix_len]
                if _dbg:
                    logger.debug(
                        "[CACHE INSERT] rid=%s freeing dup [%d:%d]=%s",
                        rid,
                        cache_protected_len,
                        new_prefix_len,
                        dup_indices[: min(len(dup_indices), 8)].tolist(),
                    )
                if dup_indices.numel() > 0:
                    runner.token_to_kv_pool_allocator.free(dup_indices)

            # --- Step 3: re-match + write-back ---
            # The tree now owns indices for [0, new_prefix_len).  Fetch them
            # and patch req_to_token_pool so the request reads the tree's
            # (still-live) indices instead of the freed ones.
            rematch = cache.match_prefix(key)
            new_indices = rematch.indices
            if _dbg:
                logger.debug(
                    "[CACHE INSERT] rid=%s rematch len=%d indices[:8]=%s",
                    rid,
                    len(new_indices),
                    new_indices[: min(len(new_indices), 8)].tolist(),
                )
            if cache.page_size == 1:
                assert len(new_indices) == seq_len, (
                    f"Re-match length mismatch after insert: "
                    f"{len(new_indices)=}, {seq_len=}, rid={rid}"
                )
            if len(new_indices) > cache_protected_len:
                if _dbg:
                    logger.debug(
                        "[CACHE INSERT] rid=%s write-back pool[slot=%d,%d:%d]=%s",
                        rid,
                        slot,
                        cache_protected_len,
                        len(new_indices),
                        new_indices[
                            cache_protected_len : cache_protected_len + 8
                        ].tolist(),
                    )
                runner.req_to_token_pool.write(
                    (slot, slice(cache_protected_len, len(new_indices))),
                    new_indices[cache_protected_len:].to(torch.int32),
                )

            # --- Step 4: update tracking ---
            self._rid_to_cache_protected_len[rid] = len(new_indices)

            # Update radix lock to cover the new (potentially deeper) node.
            old_lock = self._rid_to_radix_lock.pop(rid, None)
            if old_lock is not None:
                old_node, old_swa = old_lock
                cache.dec_lock_ref(old_node, old_swa)
            new_last_node = rematch.last_node
            if new_last_node is not None and len(new_indices) > 0:
                swa_id = cache.inc_lock_ref(new_last_node)
                self._rid_to_radix_lock[rid] = (new_last_node, swa_id)

            # --- GDN track slot association (hybrid models) ---
            if gdn_pool is not None and result.last_node is not None:
                track_slot = self._rid_to_gdn_track_slot.get(rid)
                if track_slot is not None:
                    node_id = result.last_node.id
                    old_ts = self._node_id_to_gdn_track_slot.get(node_id)
                    if old_ts is None:
                        self._node_id_to_gdn_track_slot[node_id] = track_slot
                    else:
                        gdn_pool.free_track_slot(track_slot)
                        self._rid_to_gdn_track_slot.pop(rid, None)

    # ------------------------------------------------------------------
    # KV allocation helpers
    # ------------------------------------------------------------------

    def _allocate_extend(
        self, batch: Dict[str, Any], requests_meta: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, List[int], List[int]]:
        """Allocate req pool slots and KV tokens for an extend (prefill) batch.

        Performs radix cache prefix matching before allocation:
        1. For each request, call ``match_prefix`` to find cached KV indices.
        2. Write cached indices into ``ReqToTokenPool``.
        3. Only allocate new KV tokens for the non-cached suffix.
        4. Lock matched radix nodes to prevent eviction.

        Returns ``(out_cache_loc, actual_prefix_lens, actual_extend_lens)``.
        ``out_cache_loc`` has shape ``[total_new_tokens]``.
        """
        runner = self._runner
        cache = self._radix_cache
        batch_size = batch["batch_size"]
        seq_lens: List[int] = batch["seq_lens"]

        # --- Step 1: Radix cache prefix matching ---
        actual_prefix_lens: List[int] = []
        actual_extend_lens: List[int] = []
        matched_nodes: List[Optional[TreeNode]] = []
        # Cache the match results so we don't call match_prefix twice
        cached_indices_list: List[Optional[torch.Tensor]] = []
        gdn_pool = getattr(runner, "gdn_pool", None)

        for i, m in enumerate(requests_meta):
            full_input_ids: List[int] = m.get("input_ids", [])
            full_seq_len = seq_lens[i]

            # Store input_ids for later radix cache insert
            self._rid_to_input_ids[m["rid"]] = full_input_ids

            if cache is not None and not cache.disable and len(full_input_ids) > 0:
                key = RadixKey(full_input_ids)
                match_result = cache.match_prefix(key)
                prefix_len = match_result.prefix_len
                last_node = match_result.last_node
                cached_indices = match_result.indices
            else:
                prefix_len = 0
                last_node = None
                cached_indices = None

            # Hybrid model guard: only use a KV cache hit if the matched
            # node has a GDN state snapshot.  Without it, the full-attention
            # layers would use cached KV while GDN layers start from zero,
            # causing an attention/GDN state mismatch.  Discard the hit so
            # the entire prompt is processed from scratch.
            if (
                gdn_pool is not None
                and prefix_len > 0
                and last_node is not None
                and self._node_id_to_gdn_track_slot.get(last_node.id) is None
            ):
                logger.debug(
                    "Discarding radix cache hit for rid=%s: no GDN state "
                    "for matched node (prefix_len=%d)",
                    m["rid"],
                    prefix_len,
                )
                prefix_len = 0
                last_node = None
                cached_indices = None

            # Ensure at least 2 tokens are extended (not nearly fully cached).
            # Reasons:
            # 1. A full cache hit (prefix_len == full_seq_len) would produce a
            #    0-length input tensor that crashes CUDA kernels.
            # 2. A 1-token extend triggers an edge case in FlashInfer's
            #    ragged forward_return_lse (qo_len=1, kv_len=1, causal=True)
            #    where s1 (log-partition) is computed incorrectly, causing
            #    the cascade merge to produce wrong logits → EOS.
            # By ensuring extend_len >= 2, we avoid both issues.
            if prefix_len >= full_seq_len - 1 and full_seq_len >= 2:
                prefix_len = full_seq_len - 2
                if cached_indices is not None:
                    cached_indices = cached_indices[:prefix_len]

            extend_len = full_seq_len - prefix_len
            actual_prefix_lens.append(prefix_len)
            actual_extend_lens.append(extend_len)
            matched_nodes.append(last_node)
            cached_indices_list.append(cached_indices)

            if prefix_len > 0:
                logger.info(
                    "Radix cache hit for rid=%s: %d/%d tokens reused (%.1f%%) "
                    "node_id=%s cached_kv[:8]=%s",
                    m["rid"],
                    prefix_len,
                    full_seq_len,
                    100.0 * prefix_len / full_seq_len,
                    last_node.id if last_node is not None else None,
                    cached_indices[: min(prefix_len, 8)].tolist()
                    if cached_indices is not None
                    else [],
                )
                logger.info(
                    "Radix cache tree after match: evictable=%d protected=%d",
                    cache.evictable_size(),
                    cache.protected_size(),
                )

        total_new_tokens = sum(actual_extend_lens)

        # --- Step 1.5: Lock matched radix nodes BEFORE allocation ---
        # This MUST happen before any allocation that could trigger eviction.
        # Without locking first, _alloc_kv_with_eviction could evict the
        # matched nodes, freeing their KV pool slots and causing
        # use-after-free when we later read from cached_indices.
        if cache is not None and not cache.disable:
            for i, m in enumerate(requests_meta):
                node = matched_nodes[i]
                if node is not None and actual_prefix_lens[i] > 0:
                    swa_boundary_id = cache.inc_lock_ref(node)
                    self._rid_to_radix_lock[m["rid"]] = (node, swa_boundary_id)

        # --- Step 2: Allocate req pool slots ---
        slots = runner.req_to_token_pool.alloc(batch_size)
        if slots is None:
            # Rollback locks on failure
            self._unlock_matched_nodes(requests_meta)
            raise RuntimeError("Failed to allocate req pool slots for extend batch")

        # --- Step 3: Allocate KV tokens (with eviction retry) ---
        out_cache_loc = self._alloc_kv_with_eviction(total_new_tokens)
        if out_cache_loc is None:
            for s in slots:
                runner.req_to_token_pool.free(s)
            # Rollback locks on failure
            self._unlock_matched_nodes(requests_meta)
            raise RuntimeError(
                f"Failed to allocate {total_new_tokens} KV tokens for extend batch "
                f"(even after eviction)"
            )

        # --- Step 4: Write indices into req_to_token_pool ---
        offset = 0
        for i, m in enumerate(requests_meta):
            rid = m["rid"]
            slot = slots[i]
            prefix_len = actual_prefix_lens[i]
            extend_len = actual_extend_lens[i]
            full_seq_len = seq_lens[i]

            # Write cached prefix indices (from the match result we saved)
            cached_indices = cached_indices_list[i]
            if cached_indices is not None and prefix_len > 0:
                logger.debug(
                    "[ALLOC EXTEND] rid=%s writing prefix[0:%d] to pool[slot=%d]: %s",
                    rid,
                    prefix_len,
                    slot,
                    cached_indices[: min(prefix_len, 8)].tolist(),
                )
                runner.req_to_token_pool.write(
                    (slot, slice(0, prefix_len)),
                    cached_indices[:prefix_len].to(torch.int32),
                )

            # Write new KV indices for the suffix
            kv_indices = out_cache_loc[offset : offset + extend_len]
            runner.req_to_token_pool.write(
                (slot, slice(prefix_len, full_seq_len)), kv_indices
            )

            self._rid_to_req_pool_idx[rid] = slot
            self._rid_to_kv_indices[rid] = kv_indices.clone()
            self._rid_to_output_ids[rid] = []
            # The prefix portion is already protected in the radix cache
            # (from a previous request's insert).  We start with this as
            # cache_protected_len so that subsequent insert() calls know
            # which range is already covered.
            self._rid_to_cache_protected_len[rid] = actual_prefix_lens[i]
            offset += extend_len

        # GDN state management: restore from track slot on cache hit, or reset
        if gdn_pool is not None:
            for i, m in enumerate(requests_meta):
                rid = m["rid"]
                working_slot = slots[i]
                prefix_len = actual_prefix_lens[i]
                node = matched_nodes[i]

                if prefix_len > 0 and node is not None:
                    # Cache hit — try to restore GDN state from the track slot
                    # associated with the matched radix node.
                    track_slot = self._node_id_to_gdn_track_slot.get(node.id)
                    if track_slot is not None:
                        gdn_pool.copy_states(track_slot, working_slot)
                        logger.debug(
                            "GDN state restored for rid=%s from track_slot=%d "
                            "(prefix_len=%d)",
                            rid,
                            track_slot,
                            prefix_len,
                        )
                    else:
                        # Cache hit but no GDN snapshot — reset to zero.
                        # This can happen if the track slot was evicted.
                        idx = torch.tensor(
                            [working_slot], dtype=torch.int64, device=runner.device
                        )
                        gdn_pool.reset_states(idx)
                        logger.debug(
                            "GDN state reset for rid=%s (cache hit but no "
                            "track slot, prefix_len=%d)",
                            rid,
                            prefix_len,
                        )
                else:
                    # No cache hit — fresh request, zero-init
                    idx = torch.tensor(
                        [working_slot], dtype=torch.int64, device=runner.device
                    )
                    gdn_pool.reset_states(idx)

                # Allocate a track slot only when the radix cache is enabled;
                # track slots are freed via the eviction callback so they must
                # be associated with a node, which only happens when cache is on.
                if cache is not None and not cache.disable:
                    ts = gdn_pool.alloc_track_slot()
                    if ts is not None:
                        self._rid_to_gdn_track_slot[rid] = ts

        # (Locking already done in Step 1.5 above)

        return out_cache_loc, actual_prefix_lens, actual_extend_lens

    def _unlock_matched_nodes(self, requests_meta: List[Dict[str, Any]]) -> None:
        """Rollback radix locks acquired during match_prefix.

        Called when allocation fails after locking matched nodes.
        """
        cache = self._radix_cache
        if cache is None or cache.disable:
            return
        for m in requests_meta:
            lock = self._rid_to_radix_lock.pop(m["rid"], None)
            if lock is not None:
                node, swa_id = lock
                cache.dec_lock_ref(node, swa_id)

    def _alloc_kv_with_eviction(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Try to allocate KV tokens, evicting from radix cache if needed."""
        runner = self._runner
        cache = self._radix_cache

        if num_tokens == 0:
            return torch.empty(0, dtype=torch.int32, device=runner.device)

        # First attempt: direct allocation
        result = runner.token_to_kv_pool_allocator.alloc(num_tokens)
        if result is not None:
            return result

        # Eviction loop: try evicting from radix cache to free space
        if cache is None or cache.disable:
            return None

        for attempt in range(_MAX_EVICT_RETRIES):
            evictable = cache.evictable_size()
            if evictable == 0:
                logger.warning(
                    "KV allocation failed: need %d tokens, no evictable cache entries",
                    num_tokens,
                )
                return None

            # Evict a fraction of the cache (at least what we need)
            evict_target = max(
                num_tokens,
                int(runner.token_to_kv_pool_allocator.size * _EVICT_FRACTION),
            )
            evict_result = cache.evict(evict_target)
            logger.info(
                "Radix cache eviction attempt %d: evicted %d tokens (target=%d)",
                attempt + 1,
                evict_result.full_evicted,
                evict_target,
            )

            # Retry allocation
            result = runner.token_to_kv_pool_allocator.alloc(num_tokens)
            if result is not None:
                return result

        return None

    def _allocate_decode(
        self, batch: Dict[str, Any], requests_meta: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """Allocate 1 KV token per request for a decode step.

        Returns ``out_cache_loc`` tensor of shape ``[batch_size]``.
        """
        runner = self._runner
        batch_size = batch["batch_size"]
        seq_lens: List[int] = batch["seq_lens"]

        # Allocate 1 new KV token per request (with eviction retry)
        out_cache_loc = self._alloc_kv_with_eviction(batch_size)
        if out_cache_loc is None:
            raise RuntimeError(
                f"Failed to allocate {batch_size} KV tokens for decode batch"
            )

        # Write the new KV token index into each request's mapping
        for i, m in enumerate(requests_meta):
            rid = m["rid"]
            slot = self._rid_to_req_pool_idx.get(rid)
            if slot is None:
                logger.warning("Decode step for unknown rid=%s, skipping KV write", rid)
                continue

            cur_seq_len = seq_lens[i]
            kv_new = out_cache_loc[i : i + 1]
            # The scheduler increments req.seq_len by 1 after every step, so
            # seq_lens[i] == (number of tokens in the KV cache INCLUDING the
            # token being decoded now).  The new token's slot must therefore be
            # written at index seq_lens[i] - 1, matching the position used by
            # prepare_forward_batch_decode (positions = seq_lens - 1) and the
            # window FlashInfer reads (req_to_token_pool[slot, 0:seq_lens[i]]).
            write_pos = cur_seq_len - 1
            runner.req_to_token_pool.write(
                (slot, slice(write_pos, write_pos + 1)), kv_new
            )

            # Append to tracked kv_indices
            prev = self._rid_to_kv_indices.get(rid)
            if prev is not None:
                self._rid_to_kv_indices[rid] = torch.cat([prev, kv_new])
            else:
                self._rid_to_kv_indices[rid] = kv_new.clone()

        return out_cache_loc

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def _free_rid_resources(self, rid: str) -> None:
        """Free GPU resources (req pool slot + KV indices) for a finished rid.

        KV index ownership model (when radix cache is enabled):

        ``req_to_token_pool[slot]`` contains three regions after
        ``insert()`` returns ``new_prefix_len``::

            [0, cache_protected_len)
                Indices shared with the radix tree from a previous insert.
                **Do not free** — the tree already owns them.

            [cache_protected_len, new_prefix_len)
                Indices allocated by THIS request that turned out to overlap
                with tree nodes inserted concurrently.  The tree already
                holds cloned copies → these are duplicates → **free them**.

            [new_prefix_len, total_len)
                Indices that ``insert()`` just added to the tree (cloned).
                The tree now owns the underlying KV pool slots.
                **Do not free** — the tree will free during eviction.

        When the radix cache is disabled, all KV indices are freed directly.
        """
        runner = self._runner
        cache = self._radix_cache

        slot = self._rid_to_req_pool_idx.pop(rid, None)
        kv_indices = self._rid_to_kv_indices.pop(rid, None)
        input_ids = self._rid_to_input_ids.pop(rid, None)
        output_ids = self._rid_to_output_ids.pop(rid, None)
        cache_protected_len = self._rid_to_cache_protected_len.pop(rid, 0)
        radix_lock = self._rid_to_radix_lock.pop(rid, None)
        self._rid_to_mrope_delta.pop(rid, None)

        # Free GDN track slot (if any) — the slot's association with a
        # radix node is managed separately via _node_id_to_gdn_track_slot
        # and the eviction callback; here we just remove the rid mapping.
        self._rid_to_gdn_track_slot.pop(rid, None)

        cache_enabled = cache is not None and not cache.disable

        # ----------------------------------------------------------
        # Phase 1: Read all KV indices BEFORE freeing anything.
        # ----------------------------------------------------------
        prompt_len = len(input_ids) if input_ids is not None else 0
        decode_len = len(output_ids) if output_ids else 0
        total_len = prompt_len + decode_len

        all_kv_indices: Optional[torch.Tensor] = None
        if slot is not None and input_ids is not None:
            all_kv_indices = runner.req_to_token_pool.req_to_token[slot, :total_len].to(
                torch.int64
            )

        # ----------------------------------------------------------
        # Phase 2: Insert into radix cache (if enabled).
        # ----------------------------------------------------------
        did_insert = False
        if cache_enabled and all_kv_indices is not None:
            if self._is_hybrid and decode_len > 0:
                # Hybrid model: insert only prompt tokens (not decode)
                # because GDN state is only tracked at the prompt boundary.
                prompt_kv = all_kv_indices[:prompt_len]
                decode_kv = all_kv_indices[prompt_len:]
                key = RadixKey(list(input_ids))
                result = cache.insert(key, prompt_kv)
                new_prefix_len = result.prefix_len

                # Free duplicate KV indices in the overlap region.
                if new_prefix_len > cache_protected_len:
                    dup_indices = prompt_kv[cache_protected_len:new_prefix_len]
                    if dup_indices.numel() > 0:
                        runner.token_to_kv_pool_allocator.free(dup_indices)

                # Free decode KV indices (tree does not own them)
                if decode_kv.numel() > 0:
                    runner.token_to_kv_pool_allocator.free(decode_kv)
            else:
                # Non-hybrid or no decode tokens: insert full sequence
                full_token_ids = list(input_ids)
                if output_ids:
                    full_token_ids.extend(output_ids)
                key = RadixKey(full_token_ids)
                result = cache.insert(key, all_kv_indices)
                new_prefix_len = result.prefix_len

                # Free duplicate KV indices in the overlap region.
                if new_prefix_len > cache_protected_len:
                    dup_indices = all_kv_indices[cache_protected_len:new_prefix_len]
                    if dup_indices.numel() > 0:
                        runner.token_to_kv_pool_allocator.free(dup_indices)

            did_insert = True

        # ----------------------------------------------------------
        # Phase 3: Unlock radix cache nodes.
        # ----------------------------------------------------------
        if cache_enabled and radix_lock is not None:
            node, swa_boundary_id = radix_lock
            cache.dec_lock_ref(node, swa_boundary_id)

        # ----------------------------------------------------------
        # Phase 4: Free KV indices not owned by the radix cache.
        # ----------------------------------------------------------
        if not did_insert:
            if cache_enabled and all_kv_indices is not None:
                # Cache enabled but insert skipped (shouldn't happen in
                # normal flow).  Tree owns [0, cache_protected_len);
                # free the rest.
                tail = all_kv_indices[cache_protected_len:]
                if tail.numel() > 0:
                    runner.token_to_kv_pool_allocator.free(tail)
            elif not cache_enabled:
                # Cache disabled — free all newly-allocated KV indices.
                if all_kv_indices is not None and all_kv_indices.numel() > 0:
                    runner.token_to_kv_pool_allocator.free(all_kv_indices)
                elif kv_indices is not None and kv_indices.numel() > 0:
                    runner.token_to_kv_pool_allocator.free(kv_indices)

        # ----------------------------------------------------------
        # Phase 5: Free the req pool slot.
        # ----------------------------------------------------------
        if slot is not None:
            runner.req_to_token_pool.free(slot)

        logger.debug(
            "Freed resources for rid=%s (slot=%s, kv_tokens=%d)",
            rid,
            slot,
            kv_indices.numel() if kv_indices is not None else 0,
        )

    # ------------------------------------------------------------------
    # GDN state tracking helpers (hybrid models)
    # ------------------------------------------------------------------

    def _track_gdn_state_after_extend(
        self, requests_meta: List[Dict[str, Any]]
    ) -> None:
        """Snapshot working GDN state into each request's track slot.

        Called immediately after ``runner.forward()`` for extend batches so
        that the FINAL recurrent/conv state (after processing the full prompt)
        is saved.  The track slot is later associated with a radix node in
        ``_insert_into_radix_cache``.
        """
        gdn_pool = getattr(self._runner, "gdn_pool", None)
        if gdn_pool is None:
            return

        for m in requests_meta:
            rid = m["rid"]
            working_slot = self._rid_to_req_pool_idx.get(rid)
            track_slot = self._rid_to_gdn_track_slot.get(rid)
            if working_slot is not None and track_slot is not None:
                gdn_pool.copy_states(working_slot, track_slot)

    def _on_radix_node_evict(self, node_id: int) -> None:
        """Callback invoked by RadixCache when a node is evicted.

        Frees the GDN track slot associated with the evicted node.
        """
        track_slot = self._node_id_to_gdn_track_slot.pop(node_id, None)
        if track_slot is not None:
            gdn_pool = getattr(self._runner, "gdn_pool", None)
            if gdn_pool is not None:
                gdn_pool.free_track_slot(track_slot)
                logger.debug(
                    "Freed GDN track slot %d for evicted node %d",
                    track_slot,
                    node_id,
                )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()
