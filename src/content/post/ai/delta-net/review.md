# DeltaNet 블로그 글 검토 결과

## 수학적 오류 (Critical)

### 1. Eq. 5 (line 231): Forgetting gate product 순서 오류

현재 intra-chunk 합에서 forgetting gate product가 $\beta k^T v$ **오른쪽**에 위치:

$$
\sum_{j=1}^r \left[ \beta_{[t]}^j {k_{[t]}^j}^T v_{[t]}^j \underset{\text{여기가 문제}}{\underline{\left( \prod_{i=j+1}^t (I - \beta_i k_i^T k_i) \right)}} \right]
$$

recurrence를 직접 unroll 해보면:

$$
S_2 = M_2 S_1 + X_2 = M_2(M_1 S_0 + X_1) + X_2 = M_2 M_1 S_0 + \underset{M_2가 왼쪽}{\underline{M_2 X_1}} + X_2
$$

따라서 forgetting gate product는 $\beta k^T v$ **왼쪽**에 와야 함. 행렬 곱셈은 교환법칙 성립 안 하므로 순서가 중요함.

추가로 product 상한이 `t`(chunk 인덱스)로 되어 있는데, `r`(chunk 내 위치)이어야 함.

**수정 방향**: product를 왼쪽으로 옮기고, 상한을 `r`로 변경

---

### 2. Line 276: $w_j$ 정의에서 합산 범위 오류

현재:
$$
w_j = \beta_j k_j - \beta_j k_j \sum_{m=1}^{\color{red}{t}} k_m^T w_m
$$

$\sum_{m=1}^{t}$이면 $w_j$ 자기 자신과 미래 값을 참조하게 됨 -> 순환 정의.

재귀적 정의이므로 이전에 계산된 값만 참조해야 함:
$$
w_j = \beta_j k_j - \beta_j k_j \sum_{m=1}^{\color{green}{j-1}} k_m^T w_m
$$

동일한 오류가 **3곳**에서 반복됨:
- Line 276 (WY Representation 절)
- Line 319 (Eq. 6의 $w$ 정의)
- Line 352 (마지막 섹션의 $w$ 정의)

참고로 $u$는 모든 곳에서 올바르게 $j-1$로 작성됨. $w$만 일관되게 틀림.

---

### 3. Line 190 (Eq. 3): Transpose 누락

현재:
$$
S_{[t]}^r = S_{[t]}^{0} + \sum_{j=1}^{r} k_{[t]}^{j} v_{[t]}^{j}
$$

$k \in \mathbb{R}^{1 \times d}$, $v \in \mathbb{R}^{1 \times d}$이므로 외적(outer product)을 만들려면 $k^T v$ ($d \times 1$ 곱하기 $1 \times d$ = $d \times d$)가 필요함. $kv$는 차원이 안 맞음.

수정:
$$
S_{[t]}^r = S_{[t]}^{0} + \sum_{j=1}^{r} {k_{[t]}^{j}}^T v_{[t]}^{j}
$$

Line 200에서는 올바르게 transpose 사용 중.

---

### 4. Line 297: Base case 증명 오타

현재:
```
= k_1^T (β_1 v_1) = k_1^T vu_1
```

`vu_1` -> `u_1`. 불필요한 `v`가 끼어들어감.

---

### 5. Line 94: 단위 노름(unit norm) 가정 미기재

$$
k_m S_t = v_m + \sum_{i \neq m} k_m k_i^T v_i
$$

이 식이 성립하려면 $k_m k_m^T = 1$ (단위 벡터) 가정 필요. prerequisite 글(linear-attention)의 line 95에는 명시되어 있지만, 이 글에서는 언급 없음.

**수정 방향**: "Assume $k_i$'s are normalized to unit length." 같은 문장 추가

---

## 표기법 / 일관성 문제

### 6. $O_t$ vs $o_t$ 혼용 (lines 78, 83, 110-111)

Terminology 섹션 정의:
- $o_t$: 단일 토큰 출력 벡터 (소문자)
- $O$: 전체 출력 행렬 (대문자)

그런데 recurrent form에서 단일 토큰 출력에 $O_t$ (대문자) 사용 중. $o_t$로 통일해야 함.

---

### 7. Line 44: $\hat{y}$ 표기 관례

일반적 ML 관례:
- $\hat{y}$ = 예측값 (prediction)
- $y$ = 정답 (ground truth)

이 글은 반대로 사용: $y = xW$가 예측, $\hat{y}$가 정답.

수학적으로는 내부 일관성 있으나, 독자 혼란 가능. $y^*$ 또는 $y_{\text{target}}$ 사용 권장.

---

### 8. Line 162: 헤딩 레벨 불일치

- "Reason 1" -> `###` (Parallel Scan 섹션 하위)
- "Reason 2" -> `##` (최상위 레벨)

둘 다 같은 섹션 하위이므로 "Reason 2"도 `###`이어야 함.

---

## 오타 및 문법

| Line | 현재 | 수정 |
|------|------|------|
| 44 | `ground truth  $\hat{y}$` | 이중 공백 제거 |
| 85 | `$t$ th token` | `$t$th token` (나머지와 통일) |
| 102 | `This is same form` | `This is the same form` |
| 155 | `parallel scan(assume` | `parallel scan (assume` (괄호 앞 공백) |
| 217 | `Naive Approch` | `Naive Approach` |
| 270 | `paper use column-wise` | `paper uses column-wise` |
| 462 | `a powerful properties` | `a powerful property` |
| 482 | `We gone through` | `We've gone through` |

---

## 구조적 문제

### 9. Summary 섹션 비어있음 (line 12)

`## Summary` 헤딩만 있고 내용 없음. 작성하거나 제거해야 함.

---

### 10. Mask 표기법 (lines 210, 342)

`$+ \text{Mask}$` (additive mask) 사용 중. 하지만 linear attention에는 softmax가 없으므로 $-\infty$ additive mask가 동작하지 않음.

multiplicative mask ($\odot \text{Mask}$) 사용이 맞음. prerequisite 글에서도 동일한 문제 있음.

---

## 사소한 관찰 (수정 선택사항)

- **Line 468**: $(I-A)^{-1} = I + A + \cdots + A^C$ 에서 $A$가 $C \times C$ strictly lower triangular이므로 $A^C = 0$. 급수는 $A^{C-1}$에서 끝나야 함. $A^C = 0$ 포함해도 결과는 같으나 정밀하지 않음.
- **Line 147**: parallel scan 이항 연산자 결합법칙 성립 확인 완료. 이 부분은 정확함.
- **Lines 293-307**: $S_t = \sum k_j^T u_j$ 귀납법 증명 구조 자체는 올바름 (`vu_1` 오타 제외).

---

## 요약

| 카테고리 | 개수 | 심각도 |
|----------|------|--------|
| 수학적 오류 | 5개 | Critical (1-3번), Medium (4-5번) |
| 표기법/일관성 | 3개 | Medium |
| 오타/문법 | 8개 | Low |
| 구조적 문제 | 2개 | Low-Medium |

**가장 중요한 수정 사항**: 1번(forgetting gate 순서), 2번($w_j$ 합산 범위), 3번(transpose 누락)
