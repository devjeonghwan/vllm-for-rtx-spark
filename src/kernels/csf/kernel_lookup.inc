MarlinFuncPtr get_marlin_kernel(const vllm::ScalarType a_type,
                                const vllm::ScalarType b_type,
                                const vllm::ScalarType c_type,
                                const vllm::ScalarType s_type,
                                int thread_m_blocks, int thread_n_blocks,
                                int thread_k_blocks, bool m_block_size_8,
                                bool has_zp, int group_blocks, int threads,
                                bool is_zp_float, int stages) {
  if (a_type.id() != 1125899906909960LL || b_type.id() != 562949953487106LL ||
      c_type.id() != 1125899906909960LL || s_type.id() != 2814749767172868LL ||
      has_zp || is_zp_float || stages != 4 || group_blocks != 1)
    return MarlinDefault;
  if (threads == 256 && thread_m_blocks == 3 && thread_n_blocks == 16 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[0]);
  if (threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 8 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[1]);
  if (threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
      thread_k_blocks == 8 && m_block_size_8 == true)
    return original_function(original_kernel_symbols[2]);
  if (threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 8 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[3]);
  if (threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
      thread_k_blocks == 4 && m_block_size_8 == true)
    return original_function(original_kernel_symbols[4]);
  if (threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 4 &&
      thread_k_blocks == 8 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[5]);
  if (threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 &&
      thread_k_blocks == 8 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[6]);
  if (threads == 256 && thread_m_blocks == 2 && thread_n_blocks == 16 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[7]);
  if (threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 4 &&
      thread_k_blocks == 8 && m_block_size_8 == true)
    return original_function(original_kernel_symbols[8]);
  if (threads == 256 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
      thread_k_blocks == 8 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[9]);
  if (threads == 128 && thread_m_blocks == 2 && thread_n_blocks == 4 &&
      thread_k_blocks == 8 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[10]);
  if (threads == 128 && thread_m_blocks == 3 && thread_n_blocks == 4 &&
      thread_k_blocks == 8 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[11]);
  if (threads == 256 && thread_m_blocks == 4 && thread_n_blocks == 16 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[12]);
  if (threads == 128 && thread_m_blocks == 4 && thread_n_blocks == 8 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[13]);
  if (threads == 128 && thread_m_blocks == 1 && thread_n_blocks == 8 &&
      thread_k_blocks == 4 && m_block_size_8 == false)
    return original_function(original_kernel_symbols[14]);
  return MarlinDefault;
}
