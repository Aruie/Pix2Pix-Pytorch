# Image-to-Image Translation with Conditional Adversarial Networks

- 다양한 Img2Img 문제를 일반화하여 해결할 수 있는 모델
  - 단순히 인풋을 아웃풋으로 맵핑하는것만이 아닌 손실함수 자체를 학습하여 여러가지 Task에 적용이 가능
  - Label 맵에서 이미지 생성, 외곽선 정보로 이미지 복원, Colorization 등에서 좋은 성능을 보임
- 논문 공개 이후 많은 사용자가 여러 Task에 적용한 결과 대부분 좋은 결과물을 얻음

## Related Work
- 기존방식
  - 이 방식은 한 픽셀을 결정지을때 모든 픽셀을 고려하게되어 unconstruct 하다고 함
  - Pixelwise로 Softmax 혹은 Regression을 사용한 경우가 많고 근래는 SSIM, feature matching 등의 여러가지 손실함수가 고려되고 있음
  - GAN을 사용한 방식은 있었으나 단순 맵핑에만 사용하고 L2같은 기존의 패널티만 사용했고 그결과 특정 Task에서만 사용가능한 모델이 생성됨
- cGAN이 적용된 방식
  - 출력의 joint configuration 에 패널티를 부여
  - structed loss 를 배움 이론상 타겟과 출력의 구조가 어떻게 되어있어도 패널티를 부여하는것이 가능함
- 기존 GAN 방식과의 차별점
  - Generator 에 U-Net구조를 적용
  - Discriminator 에 PatchGAN 방식 적용

# Method
- 기존의 GAN 의 Generator
  - $G :z \to y$ : latent $z$ 로부터 출력 $y$를 생성

- cGAN에서의 Generator
  - $G :\{x, z\} \to y$ : latent $z$ 만이 아닌 입력값 $x$를 받아 $y$로의 맵핑을 학습
  - 학습방식은 기존의 GAN과 동일

## 3.1 Objective
- cGAN의 손실함수
  - $\mathcal{L}_{cGAN}(G,D) = \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x,y}[\log (1-D(x,G(x,z))]$
- L1 손실함수
  - $\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y,z}[||y-G(x,z)||_1]$
- 최종 손실함수
  - $G^* = \arg \underset{G}{\min} \underset{D}{\max}  \mathcal{L}_{cGAN}(G,D) + \lambda\mathcal{L}_{L1}(G)$
- latent $z$ 의 의미
  - 여기서 latent $z$가 존재하지 않아도(기존의 접근법) y로의 맵핑은 학습이 가능하지만 deterministic한 출력을 생성한다
  - 하지만 cGAN에서 했던 결과와는 다르게 실험적으로 latent z는 큰 의미를 갖지 못하였는데 Generator 가 노이즈를 무시하는 법을 배웠기 때문으로 생각한다. (앞으로 연구가 필요한 부분)
  - 결국 Generator에 Dropout을 적용하는것으로 대체(Test 시에도 적용)하였으나 이마저 큰 변화를 주진 못하였다

## 3.2 Network Architectures
- Generator와 Discriminator 양쪽 모두 Conv - BN - ReLU 방식을 사용
- 자세한 아키텍쳐는 깃허브를 보라고함... 근데 Appendix에 어느정도 나옴
### 3.2.1 Generator with skips
- 기존 Encoder - Decoder 형식
  - img2img 문제는 대부분 고해상도의 입력과 고해상도의 출력을 맵핑하고 입력과 출력의 구조는 같음
  - 기존엔 이방식을 많이 사용하였으나 많은 translation 문제에선 입력과 출력사이의 공유되는 low-level 정보가 많아 이것을 직접 연결해줄 필요성이 있음
  - 예를들면 색상, 모서리 위치등 변하지 않는 정보들이 있음
- U-net
  - 위 방법을 개선하기 위해 i층과 n-i층을 이어주는 skip connection을 추가

### 3.2.2 Markovian discriminator (PatchGAN)
- L1 혹은 L2 로 인한 결과물은 매우 흐릿한 형상을 나타냄
  - 이는 고주파부분에는 취약하지만 저주파 부분에선 확실한 결과를 나타냄
  - 이를 이용해 discriminator는 오직 고주파부분만을 관여하게 하고 저주파 부분은 L1로스에 의존
- PatchGAN
  - 고주파 부분만 감지 하면 되므로 이미지 전체가 아닌 부분단위로 판별을 시행
  - 이미지 전체에 걸쳐 $N\times N$ 사이즈의 패치단위로 판별 후 모든 패치들의 평균을 최종 출력으로 사용
  - 사이즈에 관계없이 어떤이미지에도 적용 가능하고, 파라미터가 적어 속도도빠르며 성능도 나빠지지 않는다
  - 패치직경 이상의 거리에있는 픽셀 사이의 독립성을 가정해 효과적으로 Marcov Random Field로 모델링함
  - 

## 3.3 Optimization and Inference
- 기본적으로 GAN의 표준 학습방식인 Generator와 Discriminator를 번갈아 학습
- 추가적으로 discriminator를 학습알땐 loss 를 2로 나누어 속도를 늦춤
- Adam을 사용하였고 $lr=0.0002$, 그리고 $\beta_1=0.5, \beta_2=0.9999$ 를 사용 
- inference 에도 학습떄와 완벽히 동일한 설정 사용
  - dropout을 그대로사용
  - BatchNormalizaiton 도 학습떄의 정보가 아닌 Test시의 정보를 활용하는데 배치가 1일경우  Instance Normalization 을 의미하고 이것은 이미지 생성작업에서 효과적임이 입증되었음 (배치는 1~10을 사용)
- 이미지 입력과 출력은 단순 1 ~ 3 채널의 이미지


# 6. Appendix
## 6.1 Network Architectures
- Network
  - DCGAN의 기본 구조를 도입
  - ck : Conv-Batch-ReLU with k filter
  - cdk : Conv-Batch-Dropout-ReLU with k filter
  - 모든 Conv는 $4\times4$필터와 2 stride를 사용
  - 모든 업샘플과 다운샘플링은 2배 단위
 
 - Generator
   - encoder
     - c64 - c128 - c256 - c512 - c512 - c512 - c515 - c512
     - 첫번째 c64에는 배치노말이적용되지않음
     - encoder의 모든 ReLU는 slope 0.2를 가짐
   - decoder
     - cd512 - cd512 - cd512 - c512 - c256 - c128 - c64
     - 디코더 마지막단에 출력채널(일반적으로 3)에 맞춘 Convolution 을 Tanh와 함꼐 사용
     - decoder엔 slope가 없음
   - U-Net
     - i번째와 n-i번째 레이어에 Skip을 연결하는것 외에는 위와 모두 동일
     - 양쪽의 채널을 맞춰야 하므로 디코더의 채널수가 조금 바뀜
     - cd512 - cd1024 - cd1024 - c1024 - c1024 - c512 - c256 - c128
 - Discriminator
   - 마지막 레이어에 1차원의 출력을 갖는 컨볼루션을 사용하고 Sigmoid를 적용
   - BN은 첫번째 c64에는 적용되지 않고 모든 ReLU는 0.2의 슬로프를 가짐

   - $70\times70$ discriminator : c64 - c128 - c256 - c512
   - $1\times1$ discriminator : c64 - c128
     - 이는 좀특수한 경우로 모든 컨볼루션이 $1\times1$ 필터를 가짐
   - $16\times16$ discriminator : c64 - c128
   - $128\times128$ discriminator : c64 - c128 - c256 - c512 - c512 - c512

