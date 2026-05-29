## Model  
   
트랜스포머 블록 순서(Pre-LN 기준):  
1차(Attention 구역): LayerNorm → Attention → 잔차 더하기  
2차(FeedForward 구역): LayerNorm → FeedForward(GELU) → 잔차 더하기  
  
*각 역할에 대해 설명하자면..*  
**Attention 구역**: 단어 사이 간 맥락 및 관계를 파악함.  
**FeedForward 구역**: 개별 단어 의미를 심층적으로 학습함.  
**잔차 더하기**: output = 레이어 통과(조정값) + (통과 이전) 본래 값    
**FeedForward(GELU)**: 증폭시키고 GELU로 잔상들을 부드럽게 깎아냄  
  
난사(All-to-All) 발생 위치: Attention(문장 내 난사), FeedForward(내부 난사)  
현재로서는 글 전체에 난사를 하여 전부 파악하는 것이 <u>연산한계</u>로 인해 한정됨  
<br>  
  
**LayerNorm 구현**  
정규화 식을 사용하여 마지막 행(dim=-1)을 기준으로 정규분포 산출  
output = 증폭 × 정규화된 배열 + 편향.. 식 적용: 초기 정규화는 (1, 0) 상태  
이 두 값은 학습하며 조정이 됨  
  
여기서 GELU가 한 번 더 활성화 함수로서 필터링 역할을 함  
시간이 남는다면? → 기존 ReLU가 아닌 GEFU를 LLM에서 사용하는 이유 알아보기  
<br>  
  
**Forward**  
Linear -> GELU -> Linear.. 구조: `nn.Sequential`로 순차적 파이프라인 구축  
`nn.Linear(입력 차원, 출력 차원)`.. 예시: (5 , 12)행렬 × (12, 10 행렬) 곱이 성립 可  
"가중치 행렬"이 곱해지는데, 이 행렬은 설정된 weight 기준으로 생성됨  
  
왜 Linear에서 4배 확대되나? → 차원을 크게 부풀려야 꼬이지 않게 분석된다고 함(논문 수준)  
