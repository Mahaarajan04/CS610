SEE is 128 bit instructions for SIMD
AVX2 is 256 bit instructions for SIMD

g++ -std=c++17 -masm=att -msse4 -mavx2 -march=native -fopenmp \
     -fverbose-asm -fno-asynchronous-unwind-tables -fno-exceptions \
     -fno-rtti -fcf-protection=none -O2 problem2_modif.cpp -o problem2.out

===============================================
Averages over 5 runs: for 2^18
Serial avg: 28.20 us
OMP avg:    18.00 us
SSE avg:    18.20 us
AVX2 avg:   25.60 us
===============================================


===============================================
Averages over 5 runs: for 2^24
Serial avg: 11680.80 us
OMP avg:    11262.40 us
SSE avg:    11466.20 us
AVX2 avg:   11478.40 us
===============================================

inclusive prefix sum is a,a+b,a+b+c,...

explanation of how it works...

Perfect 😎 — let’s visualize **how your AVX2 version** computes the *inclusive prefix sum* across 8 integers at a time inside one 256-bit register.

We’ll go step-by-step so you can see exactly what happens in those lines:

```cpp
__m256i t1 = _mm256_slli_si256(x, 4);
x = _mm256_add_epi32(x, t1);

__m256i t2 = _mm256_slli_si256(x, 8);
x = _mm256_add_epi32(x, t2);
```

---

## 🧠 **1. Setup — data layout**

An AVX2 register (`__m256i`) is **256 bits = 8 × 32-bit integers**, split internally into **two 128-bit halves**:

```
Lower 128 bits:  [a0, a1, a2, a3]
Upper 128 bits:  [a4, a5, a6, a7]
```

That’s what happens after

```cpp
__m256i x = _mm256_load_si256(...);
```

---

## ⚙️ **2. Step 1 – shift by 1 element (4 bytes)**

```cpp
t1 = _mm256_slli_si256(x, 4);
x  = _mm256_add_epi32(x, t1);
```

Effect:

```
Before: [a0, a1, a2, a3, a4, a5, a6, a7]
Shifted by 4 bytes (1 int): [0, a0, a1, a2, 0, a4, a5, a6]
Add element-wise:
=> [a0, a0+a1, a1+a2, a2+a3, a4, a4+a5, a5+a6, a6+a7]
```

So now each 4-element *half* has partial pairwise sums —
we’ve added each number with its previous one.

---

## ⚙️ **3. Step 2 – shift by 2 elements (8 bytes)**

```cpp
t2 = _mm256_slli_si256(x, 8);
x  = _mm256_add_epi32(x, t2);
```

Effect:

```
Before: [a0, a0+a1, a1+a2, a2+a3, a4, a4+a5, a5+a6, a6+a7]
Shifted by 8 bytes (2 ints): [0, 0, a0, a0+a1, 0, 0, a4, a4+a5]
Add element-wise:
=> [a0,
    a0+a1,
    a0+a1+a2,
    a0+a1+a2+a3,
    a4,
    a4+a5,
    a4+a5+a6,
    a4+a5+a6+a7]
```

Now each **128-bit half** is fully inclusive scanned, *independently*.

---

## ⚠️ **4. Step 3 – fix the cross-half boundary**

AVX2 shifts (`_mm256_slli_si256`) don’t cross from the lower 128 bits into the upper 128 bits.
So currently:

```
lower half = [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3]
upper half = [a4, a4+a5, a4+a5+a6, a4+a5+a6+a7]
```

But what we really want is:

```
[ a0, a0+a1, a0+a1+a2, a0+a1+a2+a3,
  a0+a1+a2+a3+a4, ... , total up to a7 ]
```

To fix that, we add the *last element of the lower half* to all elements in the upper half:

```cpp
int carry_low = _mm256_extract_epi32(x, 3); // element 3 = last of lower half
__m256i carry_hi = _mm256_set_epi32(carry_low, carry_low, carry_low, carry_low, 0,0,0,0);
x = _mm256_add_epi32(x, carry_hi);
```

Now:

```
x = [a0,
     a0+a1,
     a0+a1+a2,
     a0+a1+a2+a3,
     a0+a1+a2+a3+a4,
     a0+a1+a2+a3+a4+a5,
     a0+a1+a2+a3+a4+a5+a6,
     a0+a1+a2+a3+a4+a5+a6+a7]
```

✅ That’s a **complete inclusive prefix sum** within one 256-bit vector!

---

## 🔁 **5. Step 4 – carry between vector blocks**

When we process the next 8 integers, we must add the *total sum of the previous block* to the new block (the `offset` in your code):

```cpp
x = _mm256_add_epi32(x, offset);
offset = _mm256_set1_epi32(_mm256_extract_epi32(x, 7)); // last element
```

This keeps the whole array cumulative across 8-element boundaries.

---

## 🎯 **6. Final result example**

Suppose `source` = `[1, 2, 3, 4, 5, 6, 7, 8]`.
Processing with AVX2 (stride=8):

| Step              | Result inside x                 | Notes              |
| ----------------- | ------------------------------- | ------------------ |
| After Step 1      | `[1, 3, 5, 7, 5, 11, 13, 15]`   | local pair sums    |
| After Step 2      | `[1, 3, 6, 10, 5, 11, 18, 26]`  | 4-wide prefix sums |
| After cross-carry | `[1, 3, 6, 10, 15, 21, 28, 36]` | full prefix sum ✅  |

---

Would you like me to show how the **SSE4 version** differs (it’s a simpler 4-lane version of this logic, no cross-lane carry needed)?
