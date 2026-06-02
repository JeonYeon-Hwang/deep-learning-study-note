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
<br>  
  
**TransformerBlock**  
```
self.norm1 = LayerNorm(normalized_shape=d_model)
x = self.norm1(x)
```
이런 식으로 "3차원 텐서"인 x를 그냥 넣어도 알아서 되는가? → broadcasting(자동 계산) 덕  
shortcut이란? → 역잔파 때 gradient가 소실되지 않도록: 통과前 + 통과後 값을 함께 전달  
<br>

**GPT Model**  
`nn.Embedding(단어 총 갯수, 임베딩 차원)`: (어찌보면?) 2차원 행렬 생성기  
구성 요소: 토큰/위치 임베딩 → 드롭아웃 → 트랜스포머 블록 × n개 → last 정규화 → 출력층  
  
pos_embeds 구성 방식:  
```
pos_indices = torch.arange(seq_len, device=idx.device)
pos_embeds = self.pos_emb(pos_indices)
```
`torch.arange(길이, gpu 등 사용여부)`: [0, 1, 2, ... n] 까지의 배열 생성  
  
*Cross Entropy Loss 구하기*  
먼저 logits(B, C_L, V_S)와 targets(B, C_L)의 차원을 맞춰야 함  
다음과 같이 차원 변환(`view`사용): logits(B × C_L, V_S) .. targets(B × C_L)  
이후 `CrossEntropyLoss()`을 통해 loss 산출  
  
*궁금한 점들..*  
단어 사전(토크나이저) 생성 과정과 임베딩 학습(트랜스포머 블록) 과정은 별개인가?..:  
.. 독립적으로 실행이 되나, 맞물려야 함(vocab_size)가 정해져야 하기 때문.  
주어진 총 단어 갯수에 비해 batch에 등장하는 단어가 적다면 train과정에서 변화는 제한적?..:  
.. batch에서 등장한 단어만 임베딩 값이 조정된다 → 역전파 과정에서 기여한 요소만 영향  
<br>  
  
**Generate Text**  
Generate 시: 매 단어가 추가될 때마다, GPT Model은 다시 순회한다(attention 다시)  
Train 시: Teacher Forcing.. 틀렸을 경우라도 강제로 답 넣고 진행  
<br>  
  
## Train  
  
Model을 훈련시키는 과정: Forward → Loss 계산 → Backward(gradient 생성) + optimize  
cross_entropy_loss 계산 방식: logits(batch 모두 합치기) -비교- targets(정답지 일렬로)  
  
`with torch.no_grad()`: 미분용 추적 graph 생성 비활성.. 기본적으로 torch는 활성 상태  
체크포인트 저장 기능: state를 저장.. model(W와 B), optimizer(M: 관성)을 각각 저장  
<br>

**Generate**
예측 단어 점수판(logits)를 통해 실제 단어 생성 로직: 실제 학습(train)이 아닌 생성용임  
temperature → 높을 수록 무작위성 증가, top_k → 점수 기준 상위 K개로 후보 한정  

`torch.multinomial(확률판, 뽑는 갯수)`: 랜덤으로 돌아가나, 확률판 기준 영역을 설정  
`torch.argmax(선택행렬, 해당행렬 차원, 차원유지 유무)`: 한 개를 max 기준 뽑음   
<br>

**train_model**  
train_loss: 內 for 문 기준, 앞서 학습한 단어 기준 loss를 산정  
val_loss: 內 for 문 기준, 새롭게 학습한 단어 기준 loss 산정  
  
대략적 함수 구성: 모델인 만큼 이 곳에 train을 총괄한다..    
```
for epoch 갯수 만큼 반복:

    for 들어온 배치, 정답지 배치 in train_loader:
        손실 점수 = calc_loss_batch(들어온 배치, 정답지 배치, ..)

        gradient 청소 => 새 loss로 미분 진행 => 가중치 수정

        if global_step이 점수 산출 주기에 도달하면?
            train_loss 점수 산출
            val_loss 점수 산출
            => 그리고 이들을 출력한다

        if global_step이 체크포인트 산출 주기에 도달하면?
            torch.save({
                key=value 형태의 저장 객체
            }, 저장 경로)
```