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