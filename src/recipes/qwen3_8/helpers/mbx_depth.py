"""Windows host for the original dynamic MTP depth policy."""
import ctypes
import json
import os
import time
import torch
from vllm.logger import init_logger

log = init_logger('vllm.mbx_depth')  # 7bb0..7cc4
from ....runtime import resources_root, cache_root
CFG = os.environ.get('MBX_DEPTH_CFG', str(resources_root() / 'recipes/qwen3_8/mtp-depth.json'))  # 6b98/7d98
STATS = os.environ.get('MBX_DEPTH_STATS', str(cache_root() / 'depth-stats.json'))  # 6bb4/7e74
LIB = os.environ['MBX_DEPTH_LIB']
STEPTIME = os.environ.get('MBX_DEPTH_STEPTIME', str(cache_root() / 'depth-steptime.json'))
_SYNCFREE_ENV = os.environ.get('MBX_DEPTH_SYNCFREE', '0') == '1'
_POLL = 4
_DYN_MODES = ('dynamic', 'opt')

# d3b4 creates LISTS, not tuples. Pair constants and outer string keys are
# direct module-init operands; these are not new fitted timing estimates.
_STEPTIME_FALLBACK = {
    '1': [[3, 1.0], [5, 1.195], [7, 1.385]],   # 823c..8370
    '4': [[3, 1.0], [5, 1.287], [7, 1.477]],   # 83a0..84d4
    '16': [[3, 1.0], [5, 1.196], [7, 1.328]],  # 8504..8630
}
_PRIOR_FALLBACK = [.85, .72, .6, .5, .42, .35, .29, .24, .2, .17, .14, .12]  # 8694..8728


class _State:
    on = False          # 9410..9434
    opt = False         # 9440..9464
    syncfree = False    # 9470..9494
    floor = ()          # 62cc/62d8 emptytuple -> 94a0
    fullset = set()     # 94c8/94f4


D = _State()            # 9578..9594/980c..9810
_mtime = [None]          # 9834..9850 -> 87b0..87c0
_L = None               # 87d8..87e4

# Windows binding admission; no replacement policy math.
from pathlib import Path
import struct
def _verify_native_pe():
    # OS/ABI admission only; original native policy math remains the pinned DLL.
    raw = Path(LIB).read_bytes()
    if len(raw) < 64 or raw[:2] != b"MZ":
        raise RuntimeError("Expected native Windows ARM64 PE")
    pe = struct.unpack_from("<I", raw, 0x3c)[0]
    if raw[pe:pe+4] != b"PE\0\0" or struct.unpack_from("<H", raw, pe+4)[0] != 0xaa64:
        raise RuntimeError("Expected native Windows ARM64 PE")



def validate_recipe_domain(config, *, world_size):
    # Explicit unsupported-domain rejection, separate from recovered math.
    if world_size != 1:
        raise NotImplementedError("This recovered host candidate admits TP1 only")
    if str(config.get('mode', 'off')).lower() != 'dynamic':
        raise NotImplementedError("Only the recovered original dynamic mode is admitted")
    if bool(config.get('log', False)):
        raise NotImplementedError("Decision log=True is outside this candidate admission")
    if config.get('floor'):
        raise NotImplementedError("Nonempty floor is outside this candidate admission")
    if not bool(config.get('syncfree', _SYNCFREE_ENV)):
        raise NotImplementedError("Original syncfree1 is required for this candidate")


def _apply(config):
    group = _tp()
    validate_recipe_domain(config, world_size=1 if group is None else group.world_size)
    return _apply_dynamic_original_region(config)

def full_depths(k):
    group = _tp()                         # 1a9cc..1aa18
    if group is None:
        config = _read_fresh()            # 1b3ec..1b438
    else:
        config = _read_fresh() if group.rank_in_group == 0 else None
        config = group.broadcast_object(config, src=0)  # 1aa68..1ab24
    if not config or str(config.get("mode", "off")).lower() not in _DYN_MODES:
        return [k]                       # 1b39c..1b3c4; does NOT clear D.fullset
    minimum = max(1, min(int(config.get("min", 3)), k))  # 1ad54..1af38
    values = list(range(minimum, k + 1))  # 1af74..1afd8
    D.fullset = set(values)               # 1affc..1b054
    return values

def full_widths(query_width):
    return [k + 1 for k in full_depths(query_width - 1)]

def depth(requested):
    if (not D.on                         # 1ed38..1edb4
            or not D.active              # 1ee34..1ee90
            or requested != D.K          # 1eed0..1ef2c; helper1e774 uses Py_NE
            or torch.cuda.is_current_stream_capturing()):  # 1ef38..1f020
        return requested                 # same input object; not int(requested)
    D.drafted = D.batch                   # 1f060..1f0d8
    return D.batch

def _tp():
    # 0x15c78..0x15d20 import/call; 0x16188 Py_GT; except at 0x15ee8.
    try:
        from vllm.distributed.parallel_state import get_tp_group
        group = get_tp_group()
        return group if group.world_size > 1 else None
    except Exception:
        return None

def _read_fresh():
    # Cached open: 0x67d8..0x67e0 -> VA 0x71c30, not state+0xc30.
    # 0x1567c open(CFG), 0x156c0 json.load, catches only these exceptions.
    try:
        return json.load(open(CFG))
    except (OSError, ValueError):
        return None

def _read():
    # _mtime is the original mutable one-item list, initially [None].
    try:
        stamp = os.stat(CFG).st_mtime
    except OSError:
        return None
    if stamp == _mtime[0]:
        return None
    # 0x21524 writes before open/JSON: do not move after successful parsing.
    _mtime[0] = stamp
    try:
        return json.load(open(CFG))
    except (OSError, ValueError) as error:
        log.warning("mbx depth: bad %s (%s), keeping the previous settings",
                    CFG, error)
        return None

def _cfg(sync):
    # Required argument named sync (0x1bd78/0x1bd84 -> VA 0x72960).
    group = _tp()
    if group is None:
        config = _read()
    elif sync:
        config = _read() if group.rank_in_group == 0 else None
        config = group.broadcast_object(config, src=0)
    else:
        return
    # Identity test, not truthiness: an empty JSON object is applied.
    if config is not None:
        _apply(config)

def _lib():
    # Exact original LP64 ctypes declaration. Windows port must change ONLY
    # the two ready arguments from c_long to c_int64 (LLP64 c_long is 32-bit).
    # LIB must point at the separately recovered native Windows ARM64 DLL.
    global _L
    if _L is None:
        _verify_native_pe()
        _L = ctypes.CDLL(LIB)
        _L.mbx_d_set.argtypes = [ctypes.c_int] * 3 + [ctypes.c_double] * 4
        _L.mbx_d_ready.argtypes = [ctypes.c_int64, ctypes.c_int64]
        _L.mbx_d_next.argtypes = [ctypes.c_int, ctypes.c_double]
        _L.mbx_d_batch.argtypes = [ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_int),
                                  ctypes.POINTER(ctypes.c_int)]
    # No explicit restype/start/window/ring argtypes in the original body.
    # Publishing _L before argtypes is deliberate original ordering.
    return _L

def _opt_seed_prior():
    # Body25390..26600; no opt/mode/active guard.
    K = D.K
    prior = (list(D.prior) + [D.prior[-1] if D.prior else 0.1] * K)[:K]
    # Original tensor call has dtype only: preserve original default-device semantics.
    D.pop_num.copy_(torch.tensor(prior, dtype=torch.float32) * 0.5)
    D.pop_den.fill_(0.5)

def _time_step(n):
    # 20148 perf_counter; 201dc subtract; 20250 setter before duration filter.
    now = time.perf_counter()
    elapsed = now - D.t_last
    D.t_last = now
    if 0.0 < elapsed < 0.5:
        value = D.st_time.setdefault(n, {}).setdefault(int(D.drafted), [0.0, 0])
        value[0] += elapsed * 1000.0
        value[1] += 1

def _stats_write():
    # Public wrapper35dd0 jumps to32f20. Guard32f98 is st_rank0, not log.
    if not D.st_rank0:
        return
    try:
        step_ms = {}
        for n, depths in sorted(D.st_time.items()):
            values = {}
            for depth, total_and_count in sorted(depths.items()):
                values[str(depth)] = [
                    round(total_and_count[0] / max(total_and_count[1], 1), 3),
                    total_and_count[1],
                ]
            step_ms[str(n)] = values
        value = {
            'steps_by_depth': D.st_hist.tolist(),
            'drafted': D.st_drafted.tolist(),
            'accepted': D.st_accepted.tolist(),
            'K': D.K,
            'max': D.max,
            'mode': D.mode,
            'step_ms': step_ms,
            't': time.time(),
        }
        temporary = STATS + '.tmp'
        with open(temporary, 'w') as stream:
            json.dump(value, stream)
        os.replace(temporary, STATS)
    except OSError:
        pass

def _set(row, chosen):
    D.depth[row] = chosen                 # 15278..152b0
    D.own[row] = chosen

def _reset(row):
    D.ptr[row] = 0                        # 16b40..16b7c
    D.last[row] = 0                       # 16bd4..16c10
    D.gen[row] += 1

def trim(scheduler_output, runner):
    # Domain: original recipe dynamic/off, D.opt=False. This function name
    # prevents confusing this proposal with a complete reconstructed module.
    # Do not turn D.on on merely to expose this function.
    if not D.on:                         # 2abf0..2ac78
        return
    request_states = getattr(runner, "req_states", None)       # 2ae60
    mapping = getattr(request_states, "req_id_to_index", None) # 2ae74
    if not D.active:                     # 2aeec..2af58
        D.batch = D.K                    # 2ba60..2baec
        D.drafted = D.K                  # 2bb60..2bb68
        return
    if mapping is not None:              # 2af60..2af64
        for request in (getattr(scheduler_output, "scheduled_new_reqs", ()) or ()):
            row = mapping.get(request.req_id)                # 2b0cc..2b104
            if row is not None:          # 2b15c..2b164; row0 is valid
                _set(row, _lib().mbx_d_start())               # 2b234..2b290
                _reset(row)              # 2b2ec..2b334
        live = [mapping[request_id] for request_id in scheduler_output.num_scheduled_tokens
                if request_id in mapping]                   # 2c5c8..2c6f0
        flag = ctypes.c_int(0)            # 2d25c..2d2dc
        depths = (ctypes.c_int * max(1, len(live)))(
            *[D.depth[row] for row in live])                 # 2d32c..2e5b8
        # IMPORTANT: n FIRST, array SECOND. Old syncfree audit prose had this
        # order wrong; actual instructions and original ctypes declaration agree.
        chosen = _lib().mbx_d_batch(len(live), depths, ctypes.byref(flag)) # 2e778..2e79c
        if D.opt:
            # Audit-domain rejection, NOT an instruction recovered from the ELF.
            # The real opt_decide/single branch is deliberately not substituted.
            raise NotImplementedError("Original opt trim branch is not recovered here")
        # The original D.opt=False branch skips opt_decide and single entirely:
        # 2e884 -> 3001c -> 30020 -> 2e9f4.
        for minimum_requests, depth_floor in D.floor:         # 2ea08..2eed8
            if len(live) >= minimum_requests:                # 2ebd0/e230 GE
                bound = D.max if D.max < depth_floor else depth_floor # 2fd10 LT
                if bound > chosen:                          # 2fda8/11610 GT
                    chosen = bound
        if D.held and not flag.value:     # 2ef18..2efbc
            for row in live:
                _reset(row)              # 2f024..2f06c
        D.held = bool(flag.value)         # 2f19c..2f248
        D.batch = chosen                 # 2f2a0..2f2a4

    D.nspec = {}                         # 2bd8c..2bdf4
    previous = D.drafted                 # 2be6c..2be78; does NOT assign drafted
    drafts = scheduler_output.scheduled_spec_decode_tokens    # 2beb0..2bebc
    if previous >= D.K or not drafts:    # 2bf50 (Py_GE), 2bf78..2bf98
        return
    D.nspec = {request_id: len(tokens) for request_id, tokens in drafts.items()} # 2c064..2c188
    D.cut_d = previous                   # 2c1d4..2c1e0
    removed = 0                         # 2c204
    for request_id, tokens in drafts.items():                 # 2c22c..2c278
        count = len(tokens)
        if count > previous:             # 2c2e8 (Py_GT)
            drafts[request_id] = tokens[:previous]            # 2e144..2e174
            scheduler_output.num_scheduled_tokens[request_id] -= count - previous # 2e198..2e23c
            removed += count - previous                      # 2e284..2e2a8
    scheduler_output.total_num_scheduled_tokens -= removed

def trim_bitmask(grammar_request_ids, bitmask):
    # Input is the original CPU NumPy row array; original structured_outputs
    # performs the async H2D copy AFTER this call. No GPU operation is added.
    if not D.on or not D.active or not D.nspec:                # 36028..36264
        return bitmask                   # 360a8..36104: SAME original object
    rows = []                            # 36268..36274
    offset = 0                           # 3627c
    cut = D.cut_d                        # 362cc..362d8
    for request_id in grammar_request_ids:                   # 36344..36388
        count = D.nspec.get(request_id, 0)                    # 363e8..36448
        # Original compare selects count on equality; no extra clamp/cast.
        kept_drafts = cut if cut < count else count           # 36510/11980 LT
        rows.extend(range(offset, offset + 1 + kept_drafts))  # 364d4..365f4
        offset += count + 1              # 36660..36680: original draft+bonus rows
    if offset != bitmask.shape[0]:       # 36bb0..36c2c/1e774 NE
        return bitmask                   # 37020: original fallback, not an accuracy pass
    if len(rows) == offset:              # 36c30..36ca4/dfa0 EQ
        return bitmask                   # no needless fancy-index copy
    return bitmask[rows]

def observe(idx_mapping, num_sampled, num_rejected):
    if not D.on:                                            # 4e2a0..4e318
        return
    D.steps += 1                                            # 4e3f4..4e4cc
    _time_step(int(num_sampled.shape[0]))                     # 4e50c..4e640
    accepted = (num_sampled.to(torch.int32) - 1).clamp(min=0)  # 4e6a8..4e7d8
    drafted = accepted + num_rejected.to(torch.int32)         # 4e870..4e8fc
    # Fixed-size statistics avoid bincount CUDA min/max readback.
    histogram_indices = drafted.clamp(max=D.K).long()
    D.st_hist.index_add_(0, histogram_indices, torch.ones_like(histogram_indices))
    D.st_drafted += (drafted[:, None] >= D.pos[None, :]).sum(0) # 4eff4..4f1c8
    D.st_accepted += (accepted[:, None] >= D.pos[None, :]).sum(0) # 4f2f0..4f404
    if D.steps % 64 == 0:                                   # 4f490..4f53c
        _cfg(sync=True)                                     # 4f5a4
        _stats_write()                                      # 4f644; log=False does not omit it
    if not D.active:                                        # 4f69c..4f6ec
        return
    rows = idx_mapping.long()                               # 4f718
    if D.opt:                                               # 4f740..4f790
        _opt_observe(rows, accepted, drafted)                # 4f7d8..4f7e4; falls through
    eligible = (drafted > 0) & (D.own[rows] <= D.drafted)     # 4f824..4fa1c
    ring_indices = D.ptr[rows] % D.ring                      # 4fa68..4fb0c
    D.acc[rows, ring_indices] = (                            # 4fb48..4fdf0
        (accepted[:, None] >= D.pos[None, :]) & eligible[:, None]
    ).to(torch.float32)
    D.ptr[rows] += eligible.long()                           # 4fe94..4ff54
    if D.steps % _POLL == 0 and not D.held:                  # 4ffb8..500ec; original _POLL=4
        if D.syncfree:                                      # 50104..5015c
            _decide_syncfree()                              # 5019c
        else:
            _decide()

def _decide():
    library = _lib()                                        # 4b4d4..4b540
    window = library.mbx_d_window()                         # 4b584..4b58c
    pointers = D.ptr.tolist()                              # 4b5ec..4b644; intentionally blocking CPU path
    for row, pointer in enumerate(pointers):                # 4b678..4b7dc
        if not library.mbx_d_ready(pointer, D.last[row]):    # 4b830..4b8b0
            continue
        old_depth = D.depth[row]                            # 4becc..4bf14
        indices = torch.tensor(                            # 4bf98..4ca04
            [(pointer - 1 - j) % D.ring for j in range(window)],
            device=D.acc.device,                           # original has no explicit dtype
        )
        rate = float(                                      # 4caa0..4cd38
            D.acc[row].index_select(0, indices)[:, old_depth - 1].sum()
        ) * 100 / window                                   # multiply THEN divide, 4cd54/4cd6c
        D.last[row] = pointer                              # 4cdd0..4ce10
        new_depth = library.mbx_d_next(old_depth, rate)      # 4ce50..4ce60
        if D.log:                                          # 4ceac..4cf00
            log.info(                                      # 4cf80..4cfa0; position is depth, not pointer
                "mbx depth · req %d · depth %d → %d · position %d accepted %.1f %% of the last %d steps",
                row, old_depth, new_depth, old_depth, rate, window,
            )
        if new_depth != old_depth:                          # 4cfec..4cffc
            _set(row, new_depth)                            # 4d9cc
            _reset(row)

def _decide_syncfree():
    library = _lib()                                        # 52a48..52ab4
    window = library.mbx_d_window()                         # 52af8..52b00
    snapshot = D.snap                                      # 52b60..52b70
    if snapshot is not None and snapshot[0].query():        # 52b9c..52c5c
        event, generations, depths = snapshot               # 52c98/52cac/52cc0
        D.snap = None                                      # 52d14..52d28; before tolist/decisions
        pointers = D.snap_ptr.tolist()                     # 52d90..52de8
        rates = D.snap_rate.tolist()                       # 52e5c..52eb4
        for row, pointer in enumerate(pointers):            # 52f68..53058
            if generations[row] != D.gen[row] or depths[row] != D.depth[row]: # 53124/53a20
                continue
            if not library.mbx_d_ready(pointer, D.last[row]): # 53aa8..53b3c
                continue
            old_depth = D.depth[row]                        # 53bc8..53c18
            rate = rates[row][old_depth - 1]                # 53c7c..53cd8
            D.last[row] = pointer                          # 53d40..53d78
            new_depth = library.mbx_d_next(old_depth, rate)  # 53dd0
            if D.log:                                      # 53e28..53e88
                log.info(                                  # 53f08..53f2c
                    "mbx depth · req %d · depth %d → %d · position %d accepted %.1f %% of the last %d steps",
                    row, old_depth, new_depth, old_depth, rate, window,
                )
            if new_depth != old_depth:                     # 53f78..53f88
                _set(row, new_depth)                       # 5674c
                _reset(row)                                # 567e0
    if D.snap is not None:                                 # 53198..53228
        return                                             # pending: do not overwrite/force sync
    ages = torch.arange(window, device=D.acc.device)        # 54110..542a4; no explicit dtype
    indices = (D.ptr[:, None] - 1 - ages[None, :]) % D.ring  # 54358..54554
    rates = D.acc.gather(                                   # 5462c..54800
        1, indices[:, :, None].expand(-1, -1, D.acc.shape[2])
    ).sum(1) * (100.0 / window)                             # 548e4..54954; divide scalar THEN multiply
    D.snap_rate.copy_(rates, non_blocking=True)              # 549dc
    D.snap_ptr.copy_(D.ptr, non_blocking=True)               # 54b40
    D.snap_ev.record()                                     # 54c28
    D.snap = (D.snap_ev, list(D.gen), list(D.depth))

def _apply_dynamic_original_region(config):
    # 373c0..385e4, original assignment/call order.
    D.mode = str(config.get('mode', 'off')).lower()
    D.log = bool(config.get('log', False))
    promote = config.get('promote', [60, 45])
    demote = config.get('demote', [25, 15])
    D.max = max(1, min(D.K, int(config.get('max', D.K))))
    _lib().mbx_d_set(int(config.get('min', 3)), D.max,
                     int(config.get('window', 48)),
                     float(promote[0]), float(promote[1]),
                     float(demote[0]), float(demote[1]))
    D.active = D.mode in _DYN_MODES
    D.opt = D.mode == 'opt'
    D.syncfree = (not D.opt) and bool(config.get('syncfree', _SYNCFREE_ENV))
    D.floor = sorted((int(n), int(d)) for n, d in (config.get('floor') or []))
    # Original3861c starts suffix='';3869c skips opt settings on non-opt.
    # This rejection is candidate-domain admission, not recovered policy math.
    if D.opt:
        raise NotImplementedError('Original opt configuration branch is not recovered')
    # 3a72c..3abc0. This log.info call is not conditional on D.log.
    log.info(
        'mbx depth: %s · min %s · max %d (K %d) · window %s · promote %s · demote %s · floor %s · sync-free %s · log %s%s',
        D.mode, config.get('min', 3), D.max, D.K, config.get('window', 48),
        promote, demote, list(D.floor), 'on' if D.syncfree else 'off',
        'on' if D.log else 'off', '',
    )

def init(speculator):
    # 41614..41698: these early checks are BEFORE the original except handler.
    K = int(speculator.num_speculative_steps)
    if K < 2:
        return
    if not os.path.exists(LIB):  # operand LIB4180c; branch41908
        return
    # Original try begins4190c before device/max_num_reqs reads41944/4196c.
    try:
        device = speculator.device
        R = int(speculator.max_num_reqs)
        D.K = K
        D.R = R
        D.steps = 0
        D.active = False
        D.log = False
        D.mode = 'off'
        D.max = K
        _cfg(sync=True)
        D.ring = _lib().mbx_d_ring()
        D.depth = [_lib().mbx_d_start()] * R
        D.own = torch.full((R,), _lib().mbx_d_start(), dtype=torch.int32, device=device)
        D.batch = K
        D.drafted = K
        D.held = False
        D.acc = torch.zeros(R, D.ring, K, dtype=torch.float32, device=device)
        D.ptr = torch.zeros(R, dtype=torch.long, device=device)
        D.last = [0] * R
        D.pos = torch.arange(1, K + 1, dtype=torch.int32, device=device)
        D.nspec = {}
        D.cut_d = K
        # Exact auxiliary subsequence42b6c..4470c, expanded, not a no-op.
        D.st_hist = torch.zeros(K + 1, dtype=torch.long, device=device)
        D.st_drafted = torch.zeros(K, dtype=torch.long, device=device)
        D.st_accepted = torch.zeros(K, dtype=torch.long, device=device)
        D.st_rank0 = _tp() is None or _tp().rank_in_group == 0
        D.st_time = {}
        D.t_last = 0.0
        D.gen = [0] * R
        D.snap = None
        D.snap_ptr = torch.zeros(R, dtype=torch.long, pin_memory=True)
        D.snap_rate = torch.zeros(R, K, dtype=torch.float32, pin_memory=True)
        D.snap_ev = torch.cuda.Event()
        D.num = torch.zeros(R, K, dtype=torch.float32, device=device)
        D.den = torch.zeros(R, K, dtype=torch.float32, device=device)
        D.pop_num = torch.zeros(K, dtype=torch.float32, device=device)
        D.pop_den = torch.zeros(K, dtype=torch.float32, device=device)
        D.opt_rate = torch.zeros(R + 1, K, dtype=torch.float32, pin_memory=True)
        D.opt_ev = torch.cuda.Event()
        D.opt_pending = False
        D.opt_gen = [0] * R
        D.opt_snap_gen = None
        D.opt_d = None
        D.opt_live = []
        if not hasattr(D, 'prior'):
            D.prior = list(_PRIOR_FALLBACK)
        _opt_seed_prior()
        D.on = True  # only after all original auxiliary allocations and seed
    except Exception as error:  # 44bb8 Exception match;44c4c fetch
        D.on = False  # 44cb4..44cc0
        log.warning('mbx depth off: %r', error)

# Not claimed: opt helpers, nonempty-floor/TP>1/logtrue support, original
# fill/note_tokens/steptime-table helper APIs unused by the pinned vLLM callsites.
# They are not supplied as no-ops. Only the seven actual pinned public consumers
# full_depths/full_widths/depth/init/observe/trim/trim_bitmask are assembled.
