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
이 각 cell의 값은 = 토큰 임베딩 + 포지셔널 임베딩(단어의 위치).. 합으로 이루어진다  
왜 합하였는데 별도의 정보가 남나?.. 직교(Orthogonal)이라는 개념이 작용한다고..  