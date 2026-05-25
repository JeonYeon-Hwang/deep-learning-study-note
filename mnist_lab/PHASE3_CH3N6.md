## ch. 3/6 중심으로 학습하기
**Network 골격 만들기**  
`class NeuralNetwork`에서 기본 골격(계층 수, 입력값 수 등)을 선언한다  

*__init__에서의 구현 방법*
```
하나의 층을 구성하는 params:
self.params = 기본 딕셔너리를 선언한다
self.params[가중치1] = np.random.randn(인풋 입력값, 노드 수) * np.sqrt(난수 정규화)
self.params[편향1] = np.zeros(노드 수)

이를 순서화된 딕셔너리에 넣는다:
self.layers = OrderedDict()
self.layers[Affine 계층1] = Affine(가중치1, 편향1)

여기에 다른 계층을 넣고 싶다면(예: ReLU):
self.layers[Relu 계층1] = ReLU() => 위에 import된 것을 끌어다 쓰기  
```
np.random.randn(인풋 입력값, 노드 수): 이는 첫째 요소 × 둘째 요소로 이루어진 행렬이다  
<br>

*forward에서의 구현 방법*
이는 비교적 간단한데, 이미 Affine 클라스에서 forward를 정의했기 때문:  
```
1층 부터 3층 까지... 쭈욱 순환 하면서 계층간 연결을 한다
for layer in self.layers.values():
    x = layer.forward(x) => forward 함수는(np.dot(x, self.W) + self.b)라 정의됨

x = self.last_layer.forward(x)
```
<br>

*backward에서의 구현 방법*  
여기선 역순으로 순회를 하면서 dout에 미분값을 반영하여 연산을 한다  
또한 gradient 지도에 각각 오차값을 반영을 하여 넣는다  
```
역순 for 루프를 가동한다 
  ├─▶ dout = layer.backward(dout)  
  │
  └─▶ 만약 지금 방 이름이 'Affine1' 이라면?
  │     ├──▶ self.grads['W1'] = 이 방 주머니에 있는 dW
  │     └──▶ self.grads['b1'] = 이 방 주머니에 있는 db
  └─▶ 다른 조건이라면 이에 맞게 분기 로직 추가
```
<br>

**BatchNorm**  
무슨 일을 하는가? → 각 레이어를 통과할 때마다 **튀는 현상**를 깎아내는 역할을 한다  
즉, 기준치(Norm)안에 들어가도록 정규화하는 역할을 수행하여 다음 레이어에 값을 전달한다  

*이해하기 위해 예시를 적어서 보자*   
레이어에서 나온 원본 값: `x = [10, 20, 30, 40 , 100]` .. 마지막 100이 튀고 있다  
BatchNorm 로직 1: 평균 구하기 → `40`  
BatchNorm 로직 2: 분산값 및 표준편차 구하기 → `1080`, `32.8`
BatchNorm 로직 3: 정규화 하기 → `x = [-0.91, -0.61, -0.30, 0.0. 1.82]`  

*왜 mean & var와 running_mean & running_var가 따로 있는가?*  
현재 구현 코드 상 train 상태라면 running_*을 "관성"을 기준으로 매번 업데이트 하나,  
비 train 상태(답 출력) 상태에서는 running_*을 그대로 사용한다  
train상태선 매번 새로운 mu/var로 정규화 하고, 실전에서는 하나의 running만 사용?  
→ running이 하나의 기준으로서 기능해야 하기 때문이라고 함(응?)  

*backward는 무슨 일을 하는가?*  
forward는 앞의 행렬 값을 정규화를 수행하였다. 그렇다면 backward는? .. 다른 back*  
처럼 오차기여도(gradient)를 만든다. 다만, dgamma, dbeta, dx라는 항목은 다름   
1. dgamma: `(넘어온 오차 × 순전파 때의 x_hat)의 열별 합`
2. dbeta: `(넘어온 오차)의 열별 합` .. 행렬의 열들의 합으로 만든 배열  
3. dx: 1단계? + 분산 오차 분담분 + 평균 오차 분담분 .. 이해 못함!
<br>

**Dropout**
dropout이 하는 일? .. 은닉층의 뉴런을 무작위로 삭제. 왜? → 과대적합을 줄이기 위해  

과대적합: 훈련 데이터에만 적응하여 실전에서 **범용적 대응**을 못하는 현상  
이를 해소하기 위해 "가중치 감소"를 한다. 이로는 불충분해서 나온 것이 "뉴런 삭제"  
<br>

### Train 구현기
Train 함수 수행 순서는 다음과 같은 주기를 반복한다:  
1. **미니배치 쪼개기**: 
   신경망을 N개의 인스턴스로 (전체/N)번 풀지 결정. 총 결과는 단일 모델에  
   평균값으로 산출되어 적용이 된다.  
2. **순전파**: 
   model에서 forward를 가져와 x_batch를 입력 → scores를 뽑아낸다.  
   해당 scores는 실제 정답과 비교하여 오차 점수를 산출.  
3. **역전파**: 
   첫 미분 산출내역(dout)을 만든 이후 이를 backward로 수행.  
   이를 통해 전체 gradient가 생성됨.  

이후에는.. gradient기반 수정 → 이를 기록하는 로직이다.
