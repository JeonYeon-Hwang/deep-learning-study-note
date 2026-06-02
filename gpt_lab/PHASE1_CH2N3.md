## Tokenization  

*토큰화와 BPE 토큰화의 차이*  
일반 토큰화는 단어를 기준으로 나누며.. 사전에 없는 단어는 인식 못함  
단어를 더 작은 단위(subword)로 쪼갬.. 조합하여 단어를 유추  
초기: 각 단어를 낱개로 쪼갬.. `안`, `녕`  
학습: 빈도 높은 단어의 낱개는 합침.. `안녕`  

*특수 토큰 정리*   
`<pad>`: 문장 길이 맞추기 위한 빈 공간용 토큰  
`<unk>`: 사전에 없는 단어용 토큰  
`<bos>`: 문장의 시작을 알리는 토큰  
`<eos>`: 문장의 끝을 알리는 토큰

*BPE 토크나이저*  
단순한 통계적 계산을 통해 단어사전을 만든다: BPE merge token  
대략적인 구현 방식은 다음과 같다:  
반복문: token pair 생성(내부 반복) → max pair 선정 → 해당 pair 기준 dict 갱신  
→ 임시 new squence 생성 → 변경된 id를 new squence에 반영(내부 반복)  
탈출조건은 두 가지다: 1. vocab size가 채워질 경우 2. sequence가 소진되었을 경우  

*save와 load는 뭐 하는 함수인가?*  
train 함수로 산출된 `id_to_token`과 `token_to_id`를 영구 저장(serialize) 및  
필요시 불러오기(deserialize)하기 위한 목적임  
모든 데이터 타입을 그대로 쓸 수 없다 → Dictionary Comprehension으로 형변환 필요!  

*encode와 decode*  
이미 `corpus.encode("utf-8")`로 id를 반환하고 있는데 encode가 필요한가?  
필요하다! → merge에서 pair로 **(id, id)=> 축약 id** 가 있기 때문에.. 이걸 밟아야 함  
왜 내 코드에서는 decode시 재귀함수 호출이 필요가 없나?.. 내 방식: `token1 + token2`  
과 같이 저장했기 때문, 통상적으론 `(token1_id, token2_id)`와 같은 튜플 형태  
<br>

## Data Set

*context length와 stride가 성능을 좌우*  
context length가 n 이라면 다음과 같은 쌍이 map에 형성된다:  
```
단어 1개 → next 단어 2번 째 
단어 2개 → next 단어 3번 째
⋮  
단어 n개 → next 단어 (n+1)번 째  
```
즉, length가 길 수록, 장문의 문맥에 대한 예측이 높아진다. 또한, stride가 작을(촘촘할)  
수록, context 단위를 적게 띄워서 쌍을 map에 만듦.. 다만, 성능과 연산량은 trade off이니 주의!  
Dataset: 무엇을 가져올 지 정의(낱개).. why? → class로 추후 `__getitem__`호출로 각각 생성  
DataLoader: 데이터 묶음(Batch)을 병렬적로 GPU에 던져줌
<br>

## Embedding

*구현되는 순서는?*  
총 토큰의 갯수 × 임베딩 차원 수 → 거대한 "랜덤값" 행렬이 생성  
이 각 cell의 (임시) 값은 = 토큰 임베딩 + 포지셔널 임베딩(단어의 위치).. 합으로 이루어진다  
왜 합하였는데 별도의 정보가 남나?.. 직교(Orthogonal)이라는 개념이 작용한다고..  

왜 임시 값인가? → 토큰 임베딩(고정값) + 반복되는 각 단어위치(포지셔널 값).. 을 하여,  
Q, K, V 변환을 하여 **Attention 연산**을 한다. 이후에 임시 값은 소멸.  

*정의하기*  
임베딩 행렬 = embedding(vocab_size, emb_dim)  
token_embedding 행렬: 전체 단어 사전, 벡터 값을 가짐  
position_embedding 행렬: context_length 만큼 행을 가짐 → 매 새 context 마다 덮어쓰기  

*forward 구현하기*  
forward에선 "이미 산출된(무작위)" 단어들의 위치값과 고유값을 더한다: 무슨 의미가 있나?  
초기엔 의미 無!.. 오차 점수 기반 학습하여 의미있는 값들로 변함  
for문 같은 순차적 방식 No, Vectorization(백터화)로 한꺼번에 연산 진행   

**간소한 trouble shooting**  
`token_extracted + position_extracted`로직에서 다음과 같은 에러 발생:  
<span style="color:red">The size of tensor a (8) must match the size of tensor b..</span>  

1. 두 3차원 배열의 각 차원 size가 매칭되지 않는다는 것으로 파악  
2. 8(a)은 sequence_length이고 128(b)은 context_length로 추정  
3. 차원 분석: 1차원(batch_size), 2차원(seq_len), 3차원(emb_dim)  
4. 다음과 같이 수정: `positions = torch.arange(x.size(1))` → x의 두 번째 차원  
<br>

## Attention  

**attention 개념**  
어느 한 문장이 있고, 각 단어마다 임베딩 백터가 있다: $E_1$, $E_2$, $E_3$..  
임베딩 벡터에 $W_Q$를 곱하면, $Q$ 쿼리 백터(질문하는 상태)가 산출된다  
행렬 계산 식: *X(1, emb_dim) × $W_Q$(emb_dim, 128) = $Q$(1, 128)*  
  
마찬가지로.. $W_K$ 를 곱해 128 차원의 $K$ 키 백터(유사도 측정)가 산출됨  
$Q$와 $K$는 Query/Key space에서 측정됨: *dot product(내적).. 큰 값 → Attend to*  
내적 값들(배열)은.. softmax를 거침, **attention pattern**이 산출됨  
  
($X × W_V$) $V$는?.. 각 단어의 핵심 정보 요소를 담음(중간 단계)  
위의 softmax($Q$와 $K$ 내적값)과 $V$가 곱해져야 특정 단어의 $\Delta E$가 나온다.  
$X$(특정 단어) + $\Delta E$ = <u>해당 문맥이 반영된 특정 단어의 의미 공간 벡터</u>가 됨  

*용어에 관한 추기 정리들..*  
Self Attention: 자기가 포함된 문장 내에서만 QKV를 만들어 문맥 파악하기  
Single-Head Attention: 128개의 차원을 한꺼번에 계산해 하나의 $\Delta E$ 산출  
Multi-Head Attention: 128개를 n개의 head로 나눔, 단어당 $\Delta E$ n개 산출  
<br>  

**구현**  
`torch.matmul(a차원 행렬, b차원 행렬)`: a!=b라도, 뒤의 2개의 차원을 곱해서 산출  
`torch.dot(1차원 행렬, 1차원 행렬)`: 오직 1차원 행렬 곱만 수행함  
`변수명.view(첫 차원, 둘째 차원... n, -1)`: 행렬을 원한는 차원으로 변환(-1: 뒤는 알아서)  
  
*multi-head는 중간 단계용일 뿐이다*  
128차원을 n개의 머리로 나눠 중간에 계산 → $\Delta E$를 구할 때 view하여 차원을 합침(128복구)  