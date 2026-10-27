# MiniMax-Sparse-Attention — Development Makefile
#
# Usage:
#   make setup                         - Install project requirements
#   make tt                            - Quick correctness smoke (1 sparse + 1 paged case)
#   make vt                            - Full sparse-attention pytest suite
#   make bb                            - End-to-end benchmark (TFLOPS output)
#   make bm                            - NCU profile around one benchmark run
#   make help                          - Show all targets

SHELL := /bin/bash
.DEFAULT_GOAL := help

PYTHON ?= python
PIP ?= pip
PYTEST ?= $(PYTHON) -m pytest

TEST_FILE := test_sparse_atten.py
PROFILE_DIR := profiles/ncu
NCU_OUT ?= $(PROFILE_DIR)/sparse_atten

# Customer benchmark defaults
B ?= 1
SQ ?= 32768
SKV ?= 32768
TOPK ?= 16
HEAD_KV ?= 4
QHEAD_PER_KV ?= 16
DIM ?= 128
BLK_KV ?= 128
WARMUP ?= 5
ITERS ?= 20
SEED ?= 42

# Benchmark toggles
BACKEND ?= cute
CUSTOMER_CASE ?= manual
Q2K_PATTERN ?= sink
PAGED ?= 0
DTYPE ?= bf16
PARTIAL_DTYPE ?= fp32
CAUSAL ?= 1
PAGE_SIZE ?= $(BLK_KV)
SEQUSED_TRIM ?= 0

# NCU profile defaults
BM_WARMUP ?= 0
BM_ITERS ?= 1
BM_ATTEN_FWD ?= 1
BM_COMBINE ?= 0
BM_PREPARE_SCHED ?= 0
NCU_CUSTOMER_CASE ?= both
NCU_Q2K_PATTERN ?= sink
NCU_SET ?= full
NCU_CLOCK_CONTROL ?= none
NCU_TARGET_PROCESSES ?= all

# NCU CLI metric set: keep this broad enough to diagnose SOL, occupancy,
# register/local-memory pressure, L2/DRAM traffic, shared-bank conflicts, and
# warp issue stalls. Append ad-hoc metrics with NCU_EXTRA_METRICS=foo,bar.
NCU_COMMA := ,
NCU_CORE_METRICS := gpu__time_duration.sum,sm__cycles_elapsed.avg,sm__cycles_elapsed.max,sm__cycles_active.avg,sm__cycles_elapsed.avg.per_second
NCU_LAUNCH_METRICS := launch__grid_size,launch__block_size,launch__thread_count,launch__waves_per_multiprocessor,launch__registers_per_thread,launch__registers_per_thread_allocated,launch__shared_mem_per_block,launch__shared_mem_per_block_static,launch__shared_mem_per_block_dynamic,launch__stack_size,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks,launch__occupancy_limit_barriers
NCU_SOL_METRICS := sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_request_throughput.avg.pct_of_peak_sustained_elapsed,gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_bytes.sum.pct_of_peak_sustained_elapsed,lts__t_requests.sum.pct_of_peak_sustained_elapsed
NCU_OCCUPANCY_METRICS := sm__warps_active.avg.pct_of_peak_sustained_elapsed,sm__maximum_warps_avg_per_active_cycle,sm__maximum_warps_per_active_cycle_pct,smsp__maximum_warps_avg_per_active_cycle,smsp__warps_active.avg,smsp__warps_active.avg.pct_of_peak_sustained_elapsed,smsp__warps_eligible.avg,smsp__warps_eligible.avg.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_elapsed
NCU_PIPE_METRICS := sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fmalite_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tc_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tma_cycles_active.avg.pct_of_peak_sustained_elapsed
NCU_INST_METRICS := sm__inst_executed.sum,sm__inst_executed_pipe_lsu.sum,sm__inst_executed_pipe_fma.sum,sm__inst_executed_pipe_xu.sum,smsp__inst_executed.sum,smsp__inst_issued.sum,smsp__thread_inst_executed.sum,smsp__sass_inst_executed_op_memory_128b.sum,smsp__sass_inst_executed_op_memory_64b.sum,smsp__sass_inst_executed_op_memory_32b.sum,smsp__sass_inst_executed_op_memory_16b.sum
NCU_MEMORY_METRICS := dram__bytes_read.sum,dram__bytes_write.sum,dram__sectors_read.sum,dram__sectors_write.sum,lts__t_bytes.sum,lts__t_requests.sum,lts__t_sector_hit_rate.pct,lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,l1tex__t_sector_hit_rate.pct,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_local_op_ld.ratio,l1tex__data_pipe_lsu_wavefronts_mem_shared.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_a.sum,l1tex__data_pipe_tc_wavefronts_mem_shared_op_utcmma_matrix_b_scope_1cta.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum,l1tex__m_l1tex2xbar_write_bytes_pipe_tma.sum,l1tex__m_l1tex2xbar_write_sectors_pipe_tma.sum
NCU_STALL_METRICS := smsp__warps_issue_stalled_barrier.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_branch_resolving.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_dispatch_stall.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_lg_throttle.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_math_pipe_throttle.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_membar.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_selected.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_tex_throttle.avg.pct_of_peak_sustained_elapsed,smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_elapsed
NCU_EXTRA_METRICS ?=
NCU_METRICS := $(NCU_CORE_METRICS),$(NCU_LAUNCH_METRICS),$(NCU_SOL_METRICS),$(NCU_OCCUPANCY_METRICS),$(NCU_PIPE_METRICS),$(NCU_INST_METRICS),$(NCU_MEMORY_METRICS),$(NCU_STALL_METRICS)$(if $(NCU_EXTRA_METRICS),$(NCU_COMMA)$(NCU_EXTRA_METRICS),)

CAUSAL_FLAG := $(if $(filter 1,$(CAUSAL)),--causal,)
PAGED_FLAG := $(if $(filter 1,$(PAGED)),--paged,)
PAGED_ARGS := $(if $(filter 1,$(PAGED)),--page-size $(PAGE_SIZE) --seqused-trim $(SEQUSED_TRIM),)
BENCH_ARGS := benchmark \
	--backend $(BACKEND) \
	--customer-case $(CUSTOMER_CASE) \
	--q2k-pattern $(Q2K_PATTERN) \
	--b $(B) \
	--sq $(SQ) \
	--skv $(SKV) \
	--topk $(TOPK) \
	--head-kv $(HEAD_KV) \
	--qhead-per-kv $(QHEAD_PER_KV) \
	--dim $(DIM) \
	--blk-kv $(BLK_KV) \
	--warmup $(WARMUP) \
	--iters $(ITERS) \
	--seed $(SEED) \
	--dtype $(DTYPE) \
	--partial-dtype $(PARTIAL_DTYPE) \
	$(PAGED_FLAG) \
	$(CAUSAL_FLAG) \
	$(PAGED_ARGS)
BM_ARGS := benchmark \
	--backend $(BACKEND) \
	--customer-case $(CUSTOMER_CASE) \
	--q2k-pattern $(Q2K_PATTERN) \
	--b $(B) \
	--sq $(SQ) \
	--skv $(SKV) \
	--topk $(TOPK) \
	--head-kv $(HEAD_KV) \
	--qhead-per-kv $(QHEAD_PER_KV) \
	--dim $(DIM) \
	--blk-kv $(BLK_KV) \
	--warmup $(BM_WARMUP) \
	--iters $(BM_ITERS) \
	--seed $(SEED) \
	--dtype $(DTYPE) \
	--partial-dtype $(PARTIAL_DTYPE) \
	--profile \
	$(PAGED_FLAG) \
	$(CAUSAL_FLAG) \
	$(PAGED_ARGS)
NCU_BM_COMMON_ARGS := benchmark \
	--customer-case $(NCU_CUSTOMER_CASE) \
	--q2k-pattern $(NCU_Q2K_PATTERN) \
	--b $(B) \
	--sq $(SQ) \
	--skv $(SKV) \
	--topk $(TOPK) \
	--head-kv $(HEAD_KV) \
	--qhead-per-kv $(QHEAD_PER_KV) \
	--dim $(DIM) \
	--blk-kv $(BLK_KV) \
	--warmup 0 \
	--iters 1 \
	--seed $(SEED) \
	--dtype $(DTYPE) \
	--partial-dtype $(PARTIAL_DTYPE) \
	--profile \
	--profile-skip-autotune \
	--sync-nvtx \
	$(CAUSAL_FLAG)

define NCU_PROFILE_RANGE
	@mkdir -p $(PROFILE_DIR)
	@OUT="$(NCU_OUT)_$(1)_$(NCU_CUSTOMER_CASE)_$(NCU_Q2K_PATTERN).%p"; \
	echo "=== NCU Profile $(1) [range: $(2)] ==="; \
	echo "=== warmup=0 iters=1, profile-skip-autotune=1 ==="; \
	echo "=== customer-case=$(NCU_CUSTOMER_CASE) q2k-pattern=$(NCU_Q2K_PATTERN) backend=$(3) ==="; \
	ncu --set $(NCU_SET) \
		--clock-control $(NCU_CLOCK_CONTROL) \
		--target-processes $(NCU_TARGET_PROCESSES) \
		--nvtx --nvtx-include "$(2)/" \
		-f -o "$$OUT" \
		$(PYTHON) -u $(TEST_FILE) $(NCU_BM_COMMON_ARGS) --backend $(3); \
	echo "Profile saved to $$OUT"
endef

QUICK_TT_SPARSE := $(TEST_FILE)::test_sparse_atten[8192-8192-8-1-2-3-True-dtype0-None-False]
QUICK_TT_PAGED := $(TEST_FILE)::test_sparse_page_atten[8192-8192-0-128-4-1-2-1-True-dtype0]

.PHONY: setup tt vt bb bm bm-cli ncu-cute ncu-kernels clean help

setup:
	@mkdir -p $(PROFILE_DIR)
	@if [ -f requirements.txt ]; then \
		echo "=== Installing requirements.txt ==="; \
		$(PIP) install -r requirements.txt; \
		if [ -f requirements-dev.txt ]; then \
			echo "=== Installing requirements-dev.txt ==="; \
			$(PIP) install -r requirements-dev.txt; \
		fi; \
	elif [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then \
		echo "=== Installing project in editable mode ==="; \
		$(PIP) install -e .; \
	else \
		echo "No dependency manifest found. Add requirements.txt or pyproject.toml/setup.py first."; \
		exit 1; \
	fi
	@echo "=== Setup complete ==="

tt:
	$(PYTEST) -q -x -s '$(QUICK_TT_SPARSE)' '$(QUICK_TT_PAGED)'

vt:
	$(PYTEST) $(TEST_FILE) -v -x -s

bb:
	$(PYTHON) -u $(TEST_FILE) $(BENCH_ARGS)

bm:
	@mkdir -p $(PROFILE_DIR)
	@NVTX=""; TAG=""; \
	if [ "$(BM_ATTEN_FWD)" = "1" ]; then \
		NVTX="$$NVTX --nvtx-include Fwd_SparseAttn_Sm100_CsrVarlen/"; TAG="$${TAG}_fwd"; \
	fi; \
	if [ "$(BM_COMBINE)" = "1" ]; then \
		NVTX="$$NVTX --nvtx-include K2_Combine/"; TAG="$${TAG}_combine"; \
	fi; \
	if [ "$(BM_PREPARE_SCHED)" = "1" ]; then \
		NVTX="$$NVTX --nvtx-include SparseAttention_PrepareScheduler/"; TAG="$${TAG}_prepare_sched"; \
	fi; \
	if [ -z "$$NVTX" ]; then \
		echo "ERROR: No kernel selected. Set at least one BM_* to 1."; exit 1; \
	fi; \
	OUT="$(NCU_OUT)$$TAG.%p"; \
	echo "=== NCU Profile [$$TAG] ($(TEST_FILE)) ==="; \
	echo "=== Tip: run 'make bb' first if you want compile cache populated ==="; \
	ncu --set full --clock-control none --nvtx $$NVTX -f -o "$$OUT" \
		$(PYTHON) -u $(TEST_FILE) $(BM_ARGS); \
	echo "Profile saved to $$OUT"

bm-cli:
	@NVTX=""; TAG=""; \
	if [ "$(BM_ATTEN_FWD)" = "1" ]; then \
		NVTX="$$NVTX --nvtx-include Fwd_SparseAttn_Sm100_CsrVarlen/"; TAG="$${TAG}_fwd"; \
	fi; \
	if [ "$(BM_COMBINE)" = "1" ]; then \
		NVTX="$$NVTX --nvtx-include K2_Combine/"; TAG="$${TAG}_combine"; \
	fi; \
	if [ "$(BM_PREPARE_SCHED)" = "1" ]; then \
		NVTX="$$NVTX --nvtx-include SparseAttention_PrepareScheduler/"; TAG="$${TAG}_prepare_sched"; \
	fi; \
	if [ -z "$$NVTX" ]; then \
		echo "ERROR: No kernel selected. Set at least one BM_* to 1."; exit 1; \
	fi; \
	echo "=== NCU CLI [$$TAG] ($(TEST_FILE)) ==="; \
	ncu --nvtx $$NVTX \
		--metrics $(NCU_METRICS) \
		--target-processes all \
		$(PYTHON) -u $(TEST_FILE) $(BM_ARGS) || true

ncu-cute:
	$(call NCU_PROFILE_RANGE,cute,cute,cute)

ncu-kernels: ncu-cute

clean:
	rm -rf $(PROFILE_DIR)
	rm -rf /tmp/$$(USER)/flash_attention_cute_dsl_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "MiniMax-Sparse-Attention — Development Targets"
	@echo ""
	@echo "  make setup                         Install project requirements"
	@echo "  make tt                            Quick smoke: 1 sparse + 1 paged pytest case"
	@echo "  make vt                            Full sparse-attention pytest suite"
	@echo "  make bb                            End-to-end benchmark via test_sparse_atten.py"
	@echo "  make bm                            NCU profile around one profile-mode benchmark run"
	@echo "  make bm-cli                        NCU CLI metrics for selected kernels"
	@echo "  make ncu-cute                      NCU profile CuTe fwd timed range"
	@echo "  make ncu-kernels                   Run CuTe NCU profiles"
	@echo "  make clean                         Clear NCU outputs and compile caches"
	@echo ""
	@echo "  Benchmark variables:"
	@echo "    PAGED=0|1           B=$(B) SQ=$(SQ) SKV=$(SKV) TOPK=$(TOPK)"
	@echo "    HEAD_KV=$(HEAD_KV) QHEAD_PER_KV=$(QHEAD_PER_KV) DIM=$(DIM) BLK_KV=$(BLK_KV)"
	@echo "    WARMUP=$(WARMUP) ITERS=$(ITERS) SEED=$(SEED) DTYPE=$(DTYPE) CAUSAL=0|1"
	@echo "    PAGE_SIZE=$(PAGE_SIZE) SEQUSED_TRIM=$(SEQUSED_TRIM)   # PAGED=1 only"
	@echo ""
	@echo "  NCU variables:"
	@echo "    BM_WARMUP=$(BM_WARMUP) BM_ITERS=$(BM_ITERS) NCU_OUT=$(NCU_OUT)"
	@echo "    BM_ATTEN_FWD=0|1 BM_COMBINE=0|1 BM_PREPARE_SCHED=0|1"
	@echo "    NCU_CUSTOMER_CASE=$(NCU_CUSTOMER_CASE) NCU_Q2K_PATTERN=$(NCU_Q2K_PATTERN)"
	@echo "    NCU_SET=$(NCU_SET) NCU_CLOCK_CONTROL=$(NCU_CLOCK_CONTROL) NCU_TARGET_PROCESSES=$(NCU_TARGET_PROCESSES)"
	@echo "    NCU_EXTRA_METRICS=metric_a,metric_b   # appended to bm-cli metrics"
	@echo ""
	@echo "  Examples:"
	@echo "    make bb"
	@echo "    make bb PAGED=1 CAUSAL=1 PAGE_SIZE=64 SEQUSED_TRIM=17"
	@echo "    make bb SQ=4096 SKV=4096 TOPK=8 HEAD_KV=2 QHEAD_PER_KV=8"
	@echo "    make bm BM_ATTEN_FWD=1"
	@echo "    make bm-cli BM_ATTEN_FWD=1"
	@echo "    make ncu-kernels NCU_CUSTOMER_CASE=both NCU_Q2K_PATTERN=sink"
	@echo "    make ncu-cute NCU_CUSTOMER_CASE=ring48k NCU_Q2K_PATTERN=uniform"
