# 강화학습을 활용한 팔레타이징 최적화

본 프로젝트는 **CJ 미래기술 챌린지** 참가를 위해 진행된 것으로, **강화학습(Deep Q-Learning)** 을 이용해 다양한 크기의 박스를 효율적으로 팔레트에 적재하는 최적화 알고리즘을 개발하는 것을 목표로 합니다.

## 문제 정의
- 다양한 크기의 박스를 주어진 팔레트에 효율적으로 적재해야 함
- 팔레트 규격, 박스 크기, 적재 안정성 등의 제약 조건을 고려
- stacking rate(적재율)를 최대화하는 것이 목표

## 접근 방법
- **상태(State)**: 현재 팔레트의 적재 상태 + 버퍼 팔레트에 적재한 N개의 박스 크기
- **행동(Action)**: 어떤 규칙으로 버퍼 팔레트에서 적재 팔레트로 상자를 옮길지 규칙 선택
- **보상(Reward)**: 적재된 면적, 박스 부피에 기반하여 설계
- **시뮬레이션 환경**: 실제 팔레타이징 과정을 모사하여 에이전트 학습

## 주요 기능
- 다양한 박스 크기 변형에 대응할 수 있도록 크기를 State에 포함
- 강화학습 기반으로 박스 크기에 따른 최적 적재 행동을 선택
- 휴리스틱 알고리즘과 비교 실험을 통해 학습 기반 접근법의 가능성을 검증

## 알고리즘 구조
![{F5115188-899D-4040-B21F-F6A7B1B2632D}](https://github.com/user-attachments/assets/7ed8363f-f5fb-4659-9af2-d954350966aa)

## 프로젝트 구조
```
CJ_palletizing_optimization/
├── first_proj/
│   ├── [코드 파일들: 강화학습 에이전트, 환경 시뮬레이터 등]
├── README.md
├── .idea/
```

## 학습 결과

![{CF472D4C-8D20-49BB-ACCC-68D1EAFD603B}](https://github.com/user-attachments/assets/47f5c4f6-5463-40c6-b575-f32028c514ab)
- reward와 실제 적재율 간의 관계를 확인
- 현재 reward로 최적해를 찾는 것에 한계가 있다는 것을 확인

![{6B6F9B4E-0ABD-41B3-B39F-FFC686FCC8D6}](https://github.com/user-attachments/assets/430ec661-3fa0-400a-afbd-43baaff50fad)
- 학습에 실패
- 해공간이 너무 넓고 이에 대한 학습 시간이 너무 모자람

## 결론 및 향후 계획
- 초기 DQN 기반 학습에서는 충분한 성능 개선이 어렵다는 문제를 분석
- 향후 **메타휴리스틱 기법** 적용 및 **다른 강화학습 알고리즘(PPO, SAC 등)** 을 활용하여 성능을 개선할 예정
- 팔레타이징 최적화에 특화된 보상 설계 및 상태 표현을 고도화할 계획


![370661](https://github.com/user-attachments/assets/db50845e-606c-422a-88ef-b8cf42a5320a)
