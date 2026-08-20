#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  // Move all keys and values to shared memory (SRAM) first
  for (size_t i = 0; i < keys.size(); ++i) {
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(values[i]);
  }

  // Precompute K and V matrices (32, 512) once
  Matrix *K_full = matrix_memory_allocator.Allocate("K_full");
  Matrix *V_full = matrix_memory_allocator.Allocate("V_full");

  // Concatenate all keys into K_full
  if (keys.size() == 1) {
    gpu_sim.Copy(keys[0], K_full, Position::kInSharedMemory);
  } else {
    Matrix *K_current = keys[0];
    for (size_t k_idx = 1; k_idx < keys.size(); ++k_idx) {
      Matrix *temp = matrix_memory_allocator.Allocate("temp_K");
      gpu_sim.Concat(K_current, keys[k_idx], temp, 0, Position::kInSharedMemory);
      K_current = temp;
    }
    gpu_sim.Copy(K_current, K_full, Position::kInSharedMemory);
  }

  // Concatenate all values into V_full
  if (values.size() == 1) {
    gpu_sim.Copy(values[0], V_full, Position::kInSharedMemory);
  } else {
    Matrix *V_current = values[0];
    for (size_t v_idx = 1; v_idx < values.size(); ++v_idx) {
      Matrix *temp = matrix_memory_allocator.Allocate("temp_V");
      gpu_sim.Concat(V_current, values[v_idx], temp, 0, Position::kInSharedMemory);
      V_current = temp;
    }
    gpu_sim.Copy(V_current, V_full, Position::kInSharedMemory);
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    size_t m = current_query->GetRowNum(); // m = i+1

    // Get first m rows of K_full → K (m, 512)
    Matrix *K = matrix_memory_allocator.Allocate("K");
    if (m == 1) {
      gpu_sim.GetRow(K_full, 0, K, Position::kInSharedMemory);
    } else {
      Matrix *K_current = matrix_memory_allocator.Allocate("K_current");
      gpu_sim.GetRow(K_full, 0, K_current, Position::kInSharedMemory);
      for (size_t r = 1; r < m; ++r) {
        Matrix *row = matrix_memory_allocator.Allocate("row");
        gpu_sim.GetRow(K_full, r, row, Position::kInSharedMemory);
        Matrix *temp = matrix_memory_allocator.Allocate("temp");
        gpu_sim.Concat(K_current, row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(K_current);
        K_current = temp;
      }
      gpu_sim.Copy(K_current, K, Position::kInSharedMemory);
    }

    // Get first m rows of V_full → V (m, 512)
    Matrix *V = matrix_memory_allocator.Allocate("V");
    if (m == 1) {
      gpu_sim.GetRow(V_full, 0, V, Position::kInSharedMemory);
    } else {
      Matrix *V_current = matrix_memory_allocator.Allocate("V_current");
      gpu_sim.GetRow(V_full, 0, V_current, Position::kInSharedMemory);
      for (size_t r = 1; r < m; ++r) {
        Matrix *row = matrix_memory_allocator.Allocate("row");
        gpu_sim.GetRow(V_full, r, row, Position::kInSharedMemory);
        Matrix *temp = matrix_memory_allocator.Allocate("temp");
        gpu_sim.Concat(V_current, row, temp, 0, Position::kInSharedMemory);
        gpu_sim.ReleaseMatrix(V_current);
        V_current = temp;
      }
      gpu_sim.Copy(V_current, V, Position::kInSharedMemory);
    }

    // Step 3: Transpose K to get K^T (512, m)
    gpu_sim.Transpose(K, Position::kInSharedMemory);

    // Step 4: Move current query to shared memory
    gpu_sim.MoveMatrixToSharedMem(current_query);

    // Step 5: Compute Q * K^T = current_query * K (shape m x m)
    Matrix *QK_T = matrix_memory_allocator.Allocate("QK_T");
    gpu_sim.MatMul(current_query, K, QK_T);

    // Step 6: Compute exp(QK_T) (element-wise)
    Matrix *exp_QK_T = matrix_memory_allocator.Allocate("exp_QK_T");
    gpu_sim.MatExp(QK_T, exp_QK_T);

    // Step 7: Compute softmax row-wise
    Matrix *softmax_result = matrix_memory_allocator.Allocate("softmax_result");
    // Process first row
    Matrix *first_row = matrix_memory_allocator.Allocate("first_row");
    gpu_sim.GetRow(exp_QK_T, 0, first_row, Position::kInSharedMemory);
    Matrix *row_sum = matrix_memory_allocator.Allocate("row_sum");
    gpu_sim.Sum(first_row, row_sum);
    Matrix *softmax_row = matrix_memory_allocator.Allocate("softmax_row");
    gpu_sim.MatDiv(first_row, row_sum, softmax_row);
    gpu_sim.Copy(softmax_row, softmax_result, Position::kInSharedMemory);
    // Release temporary matrices
    gpu_sim.ReleaseMatrix(first_row);
    gpu_sim.ReleaseMatrix(row_sum);
    gpu_sim.ReleaseMatrix(softmax_row);

    // Process remaining rows
    for (size_t r = 1; r < m; ++r) {
      Matrix *row = matrix_memory_allocator.Allocate("row");
      gpu_sim.GetRow(exp_QK_T, r, row, Position::kInSharedMemory);
      Matrix *r_sum = matrix_memory_allocator.Allocate("r_sum");
      gpu_sim.Sum(row, r_sum);
      Matrix *s_row = matrix_memory_allocator.Allocate("s_row");
      gpu_sim.MatDiv(row, r_sum, s_row);
      // Concatenate s_row to softmax_result along axis 0
      Matrix *temp = matrix_memory_allocator.Allocate("temp_softmax");
      gpu_sim.Concat(softmax_result, s_row, temp, 0, Position::kInSharedMemory);
      // Release old softmax_result
      gpu_sim.ReleaseMatrix(softmax_result);
      softmax_result = temp;
      // Release temporary matrices
      gpu_sim.ReleaseMatrix(row);
      gpu_sim.ReleaseMatrix(r_sum);
      gpu_sim.ReleaseMatrix(s_row);
    }

    // Step 8: Compute softmax_result * V (shape m x 512)
    Matrix *answer = matrix_memory_allocator.Allocate("answer");
    gpu_sim.MatMul(softmax_result, V, answer);

    // Step 9: Move answer to HBM and commit
    gpu_sim.MoveMatrixToGpuHbm(answer);
    // Release unnecessary matrices before running to save memory
    gpu_sim.ReleaseMatrix(QK_T);
    gpu_sim.ReleaseMatrix(exp_QK_T);
    gpu_sim.ReleaseMatrix(softmax_result);
    gpu_sim.ReleaseMatrix(K);
    gpu_sim.ReleaseMatrix(V);
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
