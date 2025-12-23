void cc_3d_blocked(const uint64_t* __restrict input,
                   const uint64_t (*__restrict kernel)[FIL_W][FIL_D],
                   uint64_t* __restrict result,
                   size_t OH, size_t OW, size_t OD,
                   int Bi, int Bj, int Bk) {
  for (size_t ii = 0; ii < OH; ii += Bi) {
    const size_t iMax = std::min(OH, ii + (size_t)Bi);
    for (size_t jj = 0; jj < OW; jj += Bj) {
      const size_t jMax = std::min(OW, jj + (size_t)Bj);
      for (size_t kk = 0; kk < OD; kk += Bk) {
        const size_t kMax = std::min(OD, kk + (size_t)Bk);

        for (size_t i = ii; i < iMax; ++i) {
          for (size_t j = jj; j < jMax; ++j) {
            size_t baseOut = outIdx(i, j, kk, OW, OD);
            for (size_t k = kk; k < kMax; ++k) {
              uint64_t sum = 0;
              for (size_t ki = 0; ki < FIL_H; ++ki) {
                for (size_t kj = 0; kj < FIL_W; ++kj) {
                  const uint64_t* inptr = &input[inpIdx(i + ki, j + kj, k)];
                  // small inner loop tends to vectorize well
                  for (size_t kz = 0; kz < FIL_D; ++kz) {
                    sum += inptr[kz] * kernel[ki][kj][kz];
                  }
                }
              }
              result[baseOut + (k - kk)] = sum;
            }
          }
        }
      }
    }
  }
}