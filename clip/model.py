from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# Bottleneck클래스는 ResNet의 핵심 구성 요소로, 효율적인 채널관리와 공간적 특징 추출을 통해 deep network에서 효과적으로 학습을 가능케 함, 
class Bottleneck(nn.Module): # PyTorch의 nn.Module을 상속받아 Bottleneck 클래스를 정의함, 이 클래스는 ResNet의 블록 중 하나로 사용함
    expansion = 4  # 이 값은 출력 채널수가 입력 채널수의 몇 배가 되는지를 정의함, Bottleneck 구조에서는 보통 4배로 확장됨.  
    
    def __init__(self, inplanes, planes, stride=1): # 클래스 초기화하는 메서드
        # 매개변수: \n
        # inplanes: input chanel size, 입력 텐서의 채널수 
        # planes: output chanel size (<-- 다른데서 가져온 설명), 블록 내부에서 사용할 채널수(<--chatGPT), 즉 블록내부에서 사용할 채널수가 planes가 되고 이게 곧 output 채널 사이즈가 된다.
        # 이 둘은 사이즈가 동일해야 하나? 반드시 그럴필요는 없음, 실제로 planes는 블록 내부에서 사용할 중간 채널수를 나타내고, 최종 출력 수는 planes*expansion으로 확장됌. 
        # 하지만 다음 2가지 경우에 inplanes와 planes*expansion이 동일하지 않을 수 있음
        # 1. stride가 1보다 클 때: 공간적 크기를 줄이기 위해 평균 풀링을 사용하는 경우
        # 2. 채널수를 확장할 때: planes*expansion이 inplanes와 다를때
        # 필요에 따라 두 매개변수가 다운샘플링을 통해 조정됨
        super().__init__()  # 부모클래스(nn.Module)의 초기화 메서드를 호출하여 필요한 초기화를 수행한다.

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        # 첫번째 합성곱 레이어 및 관련 레이어
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False) # 입력채널:inplanes, 출력채널: planes, 커널크기:1, 배치 정규화가 있기 때문에 편향을 사용하지 않는다.
        self.bn1 = nn.BatchNorm2d(planes)  # 첫 번째 배치 정규화 레이어, 출력 채널수는 planes가 됨.
        self.relu1 = nn.ReLU(inplace=True)  # 첫 번째 ReLU 활성화 함수, inplace=True는 메모리를 절약하기 위해 입력을 직접 수정한다.

        # 두 번째 합성곱 레이어 및 관련 레이어
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)  # 3*3 합성곱 레이어, 커널크기:3을 통해 3*3임을 알 수 있다, paddings=1: 출력 크기를 입력과 동일하게 유지하기 위해 패딩을 추가, 입력/출력채널/bias=False는 위에서 했으니 생략
        self.bn2 = nn.BatchNorm2d(planes)  # 두 번째 배치 정규화 레이어, 출력 채널수는 planes가 됨.
        self.relu2 = nn.ReLU(inplace=True)  # 두 번째 ReLU 활성화 함수,

        # 첫 번째, 두 번째 레이어로 계산한거까지 avg pooling해버린다.
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()  # 스트라이드가 1보다 클 경우평균 풀링을 사용하고, 그렇지 않으면 identity(변환 없음) 레이어를 사용한다.
        # stride가 2일 경우, 텐서의 공간적 크기를 절반(avg)으로 줄임, 
        # 이를 통해 downsampling을 수행한다.

        # 3번째 합성곱 레이어 및 관련 레이어
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)  # 입력채널은 planes, 출력채널은 planes*self.expansion을 해서 4배 확장됨, 커널은 1
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # 세번째 배치 정규화 레이어
        self.relu3 = nn.ReLU(inplace=True)  # 세번째 ReLU 활성화 함수

        self.downsample = None  # downsampling 레이어는 None으로 설정
        self.stride = stride  # 전달받은 스트라이드 값을 인스턴스 변수로 저장

        if stride > 1 or inplanes != planes * Bottleneck.expansion:  # 다운샘플링이 필요한 경우 체크한다,
            # stride가 1보다 큰 경우는 공간적 크기를 줄여야한다, 
            # 입력채널수 inplanes와 출력채널수 (planes*expansion)가 다른경우에는 채널수를 맞춰야한다
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([  # 다운샘플링레이어를 순차적으로 정의한다.
                ("-1", nn.AvgPool2d(stride)),  # 평균 풀링 레이어로 공간적 크기를 줄인다. 
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),  # 1*1 합성곱 레이어로 채널수를 확장한다.
                ("1", nn.BatchNorm2d(planes * self.expansion))  # 배치 정규화 레이어, 3번째 합성곱레이어세서 사이즈를 4배로 확장시켜준걸 그대로 사용한다. 
            ]))


    # 위에서 init함수로 세팅해준 값들을 forward함수로 전달해준다.  
    def forward(self, x: torch.Tensor):  # 순전파 메서드로, 입력 텐서 X를 받아 블록을 통과시켜 출력텐서를 반환한다. 
        identity = x  # 입력텐서를 identity 변수에 저장한다, 이는 나중에 잔차 연결에 사용된다.

        out = self.relu1(self.bn1(self.conv1(x)))  # 입력 x를 첫 번째 합성곱(conx1) -> 배치 정규화(bn1) -> ReLU활성화(relu1) 순으로 통과시킨다. 
        out = self.relu2(self.bn2(self.conv2(out)))  # 이전 단계의출력을 두번째 합성공(conv2) -> 배치 정규화(bn2) -> ReLU 활성화 (relu2) 순으로 통과시킨다. 
        out = self.avgpool(out)  # stride가 1보다 크면 평균 풀링을 수행하여 공간적 크기를 줄이고, 그렇지 않으면 변환 없이 통과시킨다. 
        out = self.bn3(self.conv3(out))  # 세 번째 합성곱 (conv3) -> 배치 정규화(bn3)를 수행한다. 

        if self.downsample is not None:  # downsampling 레이어가 존재하면, 
            identity = self.downsample(x)  # 입력 x에 다운 샘플링을 적용하여 identity를업데이트 한다.

        out += identity  # 변환된 출력 out에 원래 입력 identity를 더한다. 이는잔차 연결로, 학습을 앚넝화시키고 성능을 향상시키는 역할을 한다.
        out = self.relu3(out)  # 잔차 연결 후 ReLU 활성화 함수를 적용한다. 
        return out  # 최종 출력을 반환한다.
    


# Attention기반 풀링: multi_head_forward()를 통해 전통적인 pooling과 달리 q, k, v와 head 개수까지 중요한 공간적 특징을 선택적으로 집약한다.
# class token: Vision Transformer에서의 CLS 토큰과 유사하게, 이미지의 전반적인 특징을 대표하는 벡터를 생성한다.
# projection layer: Attention 메커니즘 내에서 Key, Query, Value를 생성하고 최종 출력을 원하는 차원으로 변환한다.
class AttentionPool2d(nn.Module):  # 2D 입력 데이터를 토큰화하고 이를 Transformer의 Attention 메커니즘을 통해 풀링하는 역할을 한다. 전통적인 pooling (avg/max pooling)과 달리, Attention을 사용하여 더 정교하게 특징을 집약한다.
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):  
        # 매개변수: \n 
        # special_dim은 입력 이미지의 공간적 차원 (가로 or 세로 크기), 예를들어 7 이면 7*7 그리드를 의미
        # embed_dim: 임베딩 차원의 수
        # num_heads: Multi-head Attention에서의 헤드 수
        # output_dim: 최종출력 임베딩 차원. 지정하지 않으면 embed_dim과 동일
        # 각 매개변수 뒤에 바로 data type을 지정해주는 문법이 type이라는 방법, 파이썬에만 가능함
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)  # 위치정보를추가하이 위한 파라미터, special_dim의 제곱 +1의 크기로 초기화된다. 추가된 +1은 클래스 토큰을 위한 공간이다.
        self.k_proj = nn.Linear(embed_dim, embed_dim)  # Key
        self.q_proj = nn.Linear(embed_dim, embed_dim)  # Query
        self.v_proj = nn.Linear(embed_dim, embed_dim)  # Value를 생성하기 위한선형변환 레이어
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)  # Attention 출력을 원하는 차원으로 변환되는 레이어
        self.num_heads = num_heads  # Multi-Head Attention에서 사용하는 헤드 수를 저장한다. 

    def forward(self, x):  
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        # 입력 x의크기가 Number(?), Channel, Height, Width이고 flatten(start_dim=2)를 통해 공간 차원 H와 W를 하나로 합쳐 [N, C, H*W]가 된다. 
        # permute(2, 0, 1)로 나온 값이 [H*W, N, C]로 변환됨, Transformer는 시퀀스 길이를 첫번째 차원으로 받기 때문임
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # x.mean(dim=0, keepdim=True): 모든 공간 토큰의 평균을 계산하여 클래스 토큰으로 사용
        # torch.cat: 클래스 토큰과 나머지 공간 토큰을 결합하여 [H*W+1, N, C] 형태로 만듦
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # positional_embedding 을 각 토큰에 더하여 위치 정보를 추가함, 여기서 위치정보라는 HWNC
        x, _ = F.multi_head_attention_forward(  # 여기서 F가 뭐지? Function의 개념
            query=x[:1], key=x, value=x,  # query는클래스 토큰(x[:1]), key 와 value는 전체 토큰 x
            embed_dim_to_check=x.shape[-1],  
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,  # 최종 선형 변환을 위한 파라미터
            out_proj_bias=self.c_proj.bias,  # 최종 선형 변환을 위한 파라미터
            use_separate_proj_weight=True,
            training=self.training, 
            need_weights=False
            # Attention을 통해 클래스 코튼에 대한 정보를 집약한다.
        )
        return x.squeeze(0)  # 클래스 토큰의 차원을 제거하여 최종 출력의 크기를 [N, C]로 만듦 


# 이 class는 ResNet의 기본 구조를 유지하면서도 CLIP의 요구 사항에 맞게 수정된 구조다, ResNet의 기본 구조는 정확히 기억이 안나지만 CLIP에서 쓰일수 있게 Attention풀링(Q, K, V을 계산)을 했다. 이는 일반적인 avg/min/max pooling보다 더 정교한 특징 추출과 집약이 가능해진다. 
# AttentionPool2D와의 관계를 봤을때 ResNet의 최종 출력 특징을 Attention 매커니즘을 통해 집약하여 이미지의 전반적인 임베딩을 생성한다.
class ModifiedResNet(nn.Module):  # ResNet의 아키텍처를 기반으로 하되, 몇 가지 수정사항을 포함, 이는 CLIP 모델의 시각적 인코더로 사용됨.
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool

    1. Stem변경: 기준 ResNet의 1개의 합성곱 대신 3개의 합성공을 사용하며, max pooling 대신 평균 풀링을사용한다.
    2. Anti-Aliasing Strided Convolution: 스트라이드가 1보다 큰 합성곱 앞에 평균 풀링을 추가하여 엘리어싱을 방지한다. 그렇담 Aliasing이 뭐지?
    3. 최종 풀링 레이어: 전통적인 평균 풀링 대신 QKV Attention 기반의 풀링을 사용한다.
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem - Stem 줄기 구성
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)  # 입력 이미지의 채널수는 3(RGB)에서 width //2 로 줄임, 커널크기 3, 스트라이드 2, 패딩 1
        self.bn1 = nn.BatchNorm2d(width // 2)  # 첫번째 배치 정규화
        self.relu1 = nn.ReLU(inplace=True)  # ReLU 활성화 함수
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)  # 두번째 배치 정규화
        self.relu2 = nn.ReLU(inplace=True)  # ReLU 활성화 함수
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)  # 세번째 배치 정규화 
        self.relu3 = nn.ReLU(inplace=True)  # ReLU 활성화 함수
        self.avgpool = nn.AvgPool2d(2)  # 커널 크기 2의평균 풀링을 통해 공간 크기를 절반으로 줄임

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        # 현재 입력 채널수를 추적함, 초기값은 width
        # self.layer1~self.layer4까지는 ResNet의 각 레이어를 생성함, 각 레이어는 여러 BottleNeck 블록으로 구성되며, 스트라이드 2는 공간크기를 줄이는 역할을 한다.
        self.layer1 = self._make_layer(width, layers[0])  
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension 
        # ResNet의 최종 출력 임베딩차원 (width*32) 형태
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        # 앞서 설명한 AttentionPool2d 클래스를사용하여 최종 특징을 집약한다, input_resoultion // 32 는 ResNet의 다운샘플링 비율(32배)를 고려한 공간차원임.


    def _make_layer(self, planes, blocks, stride=1): 
        # 매개변수 \n
        # planes: 블록 내에서 사용할 채널 수
        # blocks: 해당 레이어 내 Bottleneck 블록의 수
        # stride: 첫 번째 block의 stride
        layers = [Bottleneck(self._inplanes, planes, stride)]
        
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)
        # 이 함수의 역할은 첫 번째 Bottleneck 블록은 stride를 가지며, 입력 채널 수와 출력 채널수가 다를 수 있음, 나머지 블록은 stride 1로 동일한 채널수를 유지함.


    def forward(self, x):
        def stem(x):  # 입력 이미지를 3개의 합성곱, 배치정규화, ReLU 활성화, 평균 풀링을 거쳐 초기 특징 맵을 생성한다. 
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)  # 입력 텐서의 데이터 타입을 모델의 가중치 타입과 일치시킴, 이는 FP16과 같은 혼합 정밀도 학습에서 중요함
        x = stem(x)
        x = self.layer1(x)  # 슈슈슝~ Residual Block을 통과시킴!
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


# LayerNorm은 PyTorch의 기본 nn.LayerNor을 상속받아 FP16(Floaing Point16) 데이터 타입을 처리할 수 있도록 수정한 클래스임.
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    # FP16호환성: 입력 텐서가 FP16인 경우, LayerNorm 연산을 수행할 때 계산의 안정 성을 위해 일시적으로 FP32로 변환하여 연산을 수행하고, 다시 원래의 데이터 타입으로 변환한다. 
    # 왜 그렇게 하냐면, FP16은 FP32보다 표현 범위가 좁아 일부 연산에서 수치적 불안정성이 발생할 수 있음, 이를 방지하기 위해 연산을 FP32로 수행함.
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype  # 입력 텐서 타입 저장
        ret = super().forward(x.type(torch.float32))  # FP32로 변환 후 LayerNorm을 super()를 통해 적용
        return ret.type(orig_type)  # 다시 원래 타입으로 변환


class QuickGELU(nn.Module):  # QuickGELU는 GELU(Gausian Error Linear Unit) 활성화 함수를 빠르게 근사한 버전임, GELU는 자연스러운 비선형성을 제공하며 Transformer 계열 모델에서 널리 사용됨, QuickGELU는 근사 계산을 하는데 정확한 GELU 함수보다 계산이 간단해 속도 향상에 기여한다.
    def forward(self, x: torch.Tensor):  
        return x * torch.sigmoid(1.702 * x)
# 장점으로는 계산 효율성(시그모이드 함수는 하이브리도 방식으로 근사할 수 있어, 정확한 오류 함수보다 빠르게 계산한다)과 비슷한 성능을 유지한다.


# RAB는 Transformer의 기본 블록으로, Multi-Head Attention과 MLP(Multi-Layer Perceptron)을 포함하고 있음, 각 부분은 Residual 연결과 layer Normalization으로 구성됨.
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask:torch.Tensor = None): 
        # 매개변수: \n
        # d_model: 모델의 임베딩 차원
        # attn_mask: Attention 마스크, 주로 Casual Attention (미래 토큰을 참조하지 않도록) 등에 사용됨
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # Multi-Head Attention 레이어
        self.ln_1 = LayerNorm(d_model)  # 첫 번째 Layer Normalization 
        self.mlp = nn.Sequential(OrderedDict([  # 두 개의 선형 변환과 QuickGELU로 구성된 MLP 블록
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)  # 두 번째 Layer Normalization
        self.attn_mask = attn_mask  # Attention 마스크를 저장

    def attention(self, x: torch.Tensor):  # 입력: x는 시퀀스 길이 L과 배치 크기 N의 텐서: [L, N, d_model]
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None  # attn_mask가 있다면, 입력 텐서의 데이터 타입과 device로 변환
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]   # Multi-Head Attention을 수행하여 Attention 출력을 반환, 여기서 x가 3개인 이유가 q, k, v인 걸까? need_weights=False로 설정하여 attention 가중치 계산을 생략 (<-- 왜 생략했을까? CLIP에서 해야되기 때문에?)
        # torch.Tensor가 data type인데 x라는 변수에 저장을 함, 같은 변수에 다른 값들을 보관할수 있나? x3개에 같은 값들을 넣은건지 아니면 q,k,v를 넣은건지, 아니면 tensor자체가 dimension을 가지는 개념이니, 3차원이라는 걸 알려주는건가?

    def forward(self, x: torch.Tensor):  
        x = x + self.attention(self.ln_1(x))  # 입력 x에 LayerNorm을 적용하고 Attention을수행한 후, 원래의 x에 더함
        x = x + self.mlp(self.ln_2(x))  # 업데이트된 x에 LayerNorm을 적용하고 MLP를 수행한 후, 다시 원래의 x에 더함.
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):  
        super().__init__()
        self.width = width  # 모델의 임베딩 차원(d_model)
        self.layers = layers  # ResidualAttentionBlock의 수
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])  # ResidualAttentionBlock에서 사용하는 Attention 헤드 수
        # 모든 ResidualAttentionBlock에서 사용할 Attention 마스크
        # self.resblocks: layers수만큼의 ResidualAttentionBlock 순차적으로 쌓은 nn.Sequential 모듈

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)  # 시퀀스 텐서 X크기 ([L, N, d_model])
        # 모든 ResidualAttentionBlock을 순차적으로 통과시켜 출력 생성



# 드디어 이미지! 이미지 데이터를 Transformer 인코더로 처리하는 모델, CLIP의 이미지 인코더로 사용됨.
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        # 매개변수: \n
        # input_resolution: 입력 이미지의 해상도 (예: 224).
        # patch_size: 이미지를 패치로 나눌 때의 크기 (예: 16).
        # width: 임베딩 차원 (d_model).
        # layers: Transformer 레이어의 수.
        # heads: 각 Transformer 레이어에서의 Attention 헤드 수.
        # output_dim: 최종 출력 임베딩 차원.
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False) # conv1은 
        # conv1운 이미지 패치를 임베딩 벡터로 변환하는 합성곱 레이어.
        # 입력 채널: 3(RGB), 출력 채널: width, 커널 크기: patch_size, 스트라이드: patch_size (패치를 겹치지 않도록 분할).


        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))  # 클래스 토큰으로 사용될 학습 가능한 파라미터. 크기 width.
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))  # 패치의 위치 정보를 담은 임베딩. 크기 ((input_resolution // patch_size) ** 2 + 1, width).
        self.ln_pre = LayerNorm(width)  # 패치 임베딩 전에 적용할 LayerNorm.

        self.transformer = Transformer(width, layers, heads)  # 앞서 설명한 Transformer 클래스를 사용하여 패치 임베딩을 변환.

        self.ln_post = LayerNorm(width)  # Transformer 출력에 적용할 LayerNorm.
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))  # Transformer 출력 (width)을 output_dim으로 변환하는 선형 변환 파라미터.

    def forward(self, x: torch.Tensor):
        # 1. 패치임베딩
        x = self.conv1(x)  # shape = [N, width, grid, grid], 이미지 x를 패치로분할하여 임베딩
        # 2. 패치 텐서 재구성
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [N, width, grid ** 2] 
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 3. 클래스 토큰 추가
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [N, grid ** 2 + 1, width], 클래스 토큰을 앞에추가
        # 4. 클래스 토큰을 앞에 추가하여 [N, grid*grid+1, width]
        x = x + self.positional_embedding.to(x.dtype) # 패치의위치 정보를더함
        # 5. Layer Normalization
        x = self.ln_pre(x)  # 패치 임베딩에 LayerNorm 적용
        # 6. Transformer 인코더 통과
        x = x.permute(1, 0, 2)  # NLD -> LND, [L, N, Width] 형태로 변환
        x = self.transformer(x)  # Transformer 인코더를 통과하여 변환된 특징 추출 
        x = x.permute(1, 0, 2)  # LND -> NLD, [N, L, Width] 형태로 변환 
        # 7. 최종 정규화 및 프로젝션
        x = self.ln_post(x[:, 0, :])  # 2번째 Layer Normalization, 클래스 코튼에 LayerNorm 적용

        if self.proj is not None: # 최종 임베딩 차원으로 선형 변환
            x = x @ self.proj  

        return x



# 드디어 CLIp!! 이미지와 텍스트를 동시에 인코딩하여 벡터 공간에서의 유사성을 계산하는 모델, 이는 이미지-텍스트 쌍의 학습을 통해 다양한 멀티모달 작업을 수행할 수 있게 한다.
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        
        # 매개변수: \n
        # embed_dim: 최종 임베딩 차원.
        # Vision 관련:
            # image_resolution: 입력 이미지 해상도.
            # vision_layers: Vision 인코더의 레이어 구성 (ResNet의 경우 각 블록의 Bottleneck 수, Vision Transformer의 경우 Transformer 레이어 수).
            # vision_width: Vision 인코더의 임베딩 차원.
            # vision_patch_size: Vision Transformer에서의 패치 크기.
        # Text 관련:
            # context_length: 텍스트의 최대 토큰 수.
            # vocab_size: 단어 집합의 크기.
            # transformer_width: 텍스트 Transformer의 임베딩 차원.
            # transformer_heads: 텍스트 Transformer의 Attention 헤드 수.
            # transformer_layers: 텍스트 Transformer의 레이어 수.
        super().__init__()

        self.context_length = context_length

        # Vision 인코더 선택
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64  # vision_layers가 tuple/list인 경우, ModifiedREsNet을 사용
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,  # vision_width*32 //64 = vision_width/2
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64  # vision_heads계산
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # Transformer 인코더
        self.transformer = Transformer(  # 텍스트 인코더로 사용할 Transformer를 초기화한다. 
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()  # Casual Attention을 위한 마스크
        )  # 텍스트 인코더로 사용할 Transformer를 초기화함, attn_mask: Causal Attention을 위한 마스크.

        # 텍스트 임베딩 및 정규화
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  # 단어 임베딩 레이어
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))  # 단어의 위치 정보를 담은 학습 가능한 파라미터
        self.ln_final = LayerNorm(transformer_width) # 텍스트 인코더의 마지막 LayerNorm

        # 텍스트 프로젝션 및 Logit scale
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))  # 텍스트 인코더의 출력을 이미지 임베딩과 동일한 차원(embed_dim)으로 변환하는 선형 변환 파라미터
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 유사도 계산 시 사용되는 스케일 파라미터. 초기값은 log(1 / 0.07)

        self.initialize_parameters()  # 모델의 가중치를 초기화하는 메서드를 호출한다.

    # 파라미터 초기화 메서드
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)  # 정규 분포로 초기화, 왜 0.02일까..?
        nn.init.normal_(self.positional_embedding, std=0.01)  # 정규 분포로 초기화, 왜 0.01일까..?

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5  # Q, K, V, C 프로젝션 레이어의 가중치를 정규 분포로 초기화
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)  
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std) 
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)  # 모든 ResNet 블록에서 bn3.weight 파라미터를 0으로 초기화. 이는 Residual 블록의 출력이 초기에는 입력과 동일하게 유지되도록 도와줌.

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5  
        fc_std = (2 * self.transformer.width) ** -0.5
        # attn_std, proj_std, fc_std로 설정된 표준편차를 사용하여 가중치를 정규 분포로 **-0.5 초기화.
        for block in self.transformer.resblocks:  
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


    # Attention Mask 생성: 목적은 Casual Attention 마스크를 생성하여 텍스트 인코더에서 현재 토큰 이후의 토큰을 참조하지 못하도록 함. --> 양심상 무슨말인지 모르겠어.
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)  # 상삼각 행렬을 생성하고  # 상삼각이 뭐지?
        mask.fill_(float("-inf"))  # 상삼각 부분을 -inf로 채움
        # 이는 Attention 계산 시, -inf가 들어간 위치는 무시되도록 하는데...근데 그럼 안좋은거 아냐?
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    # dtype 속성, 모델의 파라미터 데이터 타입을 반환함, 이는 입력 텐서의 타입과 일치시켜 혼합 정밀도 학습을 지원함.
    @property
    def dtype(self):  
        return self.visual.conv1.weight.dtype

    # 이미지 인코딩 메서드
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))
    # 입력: 이미지 텐서
    # 동작: 시각적 인코더(self.visual)를 통해 이미지 특징을 추출함
    # 출력: 이미지 임베딩 벡터

    # 텍스트 인코딩 메서드
    def encode_text(self, text):  
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # 텍스트 토큰을 임베딩 벡터로 변환, [batch_size, n_ctx, transformer_width]
        x = x + self.positional_embedding.type(self.dtype)  # positional embedding, 위치 정보를 추가. 여기서 위치란게 뭘까..? 이 정보를 추가하는게 뭐가 중요한걸까
        # Transformer 인코더 통과
        x = x.permute(1, 0, 2)  # NLD -> LND, [n_ctx, batch_size, transformer_width] 형태로 변환
        x = self.transformer(x)  # Transformer 인코더를 통과
        x = x.permute(1, 0, 2)  # LND -> NLD, [batch_size, n_ctx, transformer_width] 형태로 변환
        x = self.ln_final(x).type(self.dtype)  # 최종 LayerNorm 적용

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # 우와 이런 코드 처음봄, 
        # 각 시퀀스에서 EOT(End of Text) 토큰의 위치를 찾는다.
        # 토큰의 특징을 text_projection을 통해 이미지 임베딩과 동일한 차원으로 변환

        return x

    # 순전파 메서드
    def forward(self, image, text):
        image_features = self.encode_image(image)  # 시각적 인코더를 통해 이미지 임베딩 추출
        text_features = self.encode_text(text)  # 텍스트 인코더를 통해 텍스트 임베딩 추출

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True) 
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()  # 학습 가능한 스케일파라미터를 지수 함수로 변환하여 사용
        logits_per_image = logit_scale * image_features @ text_features.t()  # 이미지 임베딩과 텍스트 임베딩 간의 코사인 유사도를 스케일링해서 계산
        logits_per_text = logits_per_image.t()  # logits_per_image의 전치 행렬

        # shape = [global_batch_size, global_batch_size] <--크기의 로짓 행렬을 반환함, 이는이미지와 텍스트 간의 유사도를 나타냄
        return logits_per_image, logits_per_text
    # 여기서!
    # 1. 멀티모달 학습: 이미지와 텍스트를 동시에 인코딩하여 공통 벡터 공간에서의 유사성을 학습함
    # 2. Contrastive Learning: 양자간의 유사도를 최대화하고, 서로 다른 쌍의 유사도를 최소화하여 효과적인 특징을 학습한다.
    # 3. 유연한 인코더 선택: ResNet 또는 ViT 중 선택하여 시각적 인코더를 구성 할 수 있다.


# convert_weights 함수는 모델의 일부 파라미터를 FP16으로 변환하여 메모리 사용량을 줄이고 계산 속도를 향상시키는 역할을 함.
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):  # Convolution 및 Linear 레이어, 파라미터 nn.Conv1d, nn.Conv2d, nn.Linear의 가중치(weight)와 편향(bias)을 FP16으로 변환
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:  # Fp16으로 변환
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:  # 이들 파라미터가 존재하면 FP16으로 변환, 이는 모델의 나머지 부분과 호환성을 유지하는데 도움된다.
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)  # 모델의 모든 서브모듈에 대해 _convert_weights_to_fp16 함수를 재귀적으로 적용한다.
# 혼합 정밀도 학습 지원: 모델의 일부 파라미터를 FP16으로 변환하여 메모리사용량을 줄이고, GPU에서의 계산 속도를 향상시킨다.
# 효율성: FP16은 FP32보다 절반의 메모리를 사용하며, 많은 GPU에서 더 빠른 연산을 지원한다.
# 호한성: 일부 파라미터만 FP16으로 변환하여 모델의 나머지 부분과 호환성을 유지한다.


# 저장된 상태 사전을 로드하여 CLIP 모델을 구축하고 초기화함
def build_model(state_dict: dict):
    # 1. vision 인코더 유형을 결정
    vit = "visual.proj" in state_dict   

    # 2. Vision 인코더 파라미터 추출:
    if vit:  # 2-1) state_dict에 "visual.proj" 키가 있으면 Vision Transformer(ViT)를 사용
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # conv1 레이어의출력 채널 수
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])  # ViT Attention 레이어 수 
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # conv1 레이어의 커널크기
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)  # positional 임베딩의 grid 크기, (positional_embedding.shape[0]-1)의 제곱근
        image_resolution = vision_patch_size * grid_size  # 패치 크기와 그리드 크기를 곱해서 이미지 해상도 계산
    else:  # 2-2) 그렇지 않으면 ModifiedResNet을 사용
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)  # 각 레이어의 블록 수를 tuple로 저장
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # layer1의 첫번째 conv1 레이어의 출력수
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)  # Attention Pooling의 positional embedding grid크기
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32  # (ResNet의 다운샘플링 비율 고려)

    # 3. 텍스트 인코더 파라미터 추출
    embed_dim = state_dict["text_projection"].shape[1]  # 텍스트 프로젝션의 출력차원
    context_length = state_dict["positional_embedding"].shape[0]  # 텍스트의 최대 토큰 수
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 단어 집합의 크기
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 텍스트 Transformer의 임베딩 차원
    transformer_heads = transformer_width // 64  # Attention 헤드 수
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    # 텍스트 Transformer의 레이어 수


    # 4. CLIP 모델 인스턴스화
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 5. 불필요한 키 제거 
    for key in ["input_resolution", "context_length", "vocab_size"]:   # 어떻게하다가 이 key들이 불필요하게 됐을까
        if key in state_dict:
            del state_dict[key]


    # 6. 파라미터 변환 및 로딩
    convert_weights(model)  # FP16으로 변환
    model.load_state_dict(state_dict)  # 상태 사전을 모델에 로드
    return model.eval()  # 평가 모드로 전환


# 전체 흐름
# 1. 데이터 입력:
# 이미지: 시각적 인코더(ModifiedResNet 또는 VisionTransformer)를 통해 이미지 임베딩으로 변환.
# 텍스트: 텍스트 인코더(Transformer)를 통해 텍스트 임베딩으로 변환.

#2. 특징 추출:
# 시각적 인코더: 이미지의 고차원 특징을 추출하여 고정된 크기의 임베딩 벡터를 생성.
# 텍스트 인코더: 텍스트의 의미적 특징을 추출하여 고정된 크기의 임베딩 벡터를 생성.

# 3. 유사도 계산:
# 코사인 유사도: 이미지 임베딩과 텍스트 임베딩 간의 유사도를 계산하여, 이미지-텍스트 쌍의 일치 정도를 평가.

# 4. 모델 학습 및 평가:
# Contrastive Learning: 이미지-텍스트 쌍의 유사도를 최대화하고, 비일치 쌍의 유사도를 최소화하여 모델을 학습.
# 평가: 모델은 다양한 멀티모달 작업(예: 이미지 캡셔닝, 텍스트 기반 이미지 검색 등)에 사용될 수 있습니다.




# 각 func의 역할
# CLIP 모델의 핵심 구성 요소들을 상세히 구현한 것으로, 이미지와 텍스트를 동시에 인코딩하여 공통 벡터 공간에서의 유사성을 학습하는 구조를 가지고 있다. 
# 각 클래스와 함수는 모델의 특정 부분을 담당하며, 전체적으로 효과적인 멀티모달 학습을 지원한다.
# AttentionPool2d: 이미지 특징을 Attention을 통해 집약.
# ModifiedResNet: ResNet을 기반으로 한 시각적 인코더.
# VisionTransformer: Transformer 기반의 시각적 인코더.
# ResidualAttentionBlock: Transformer의 기본 블록.
# Transformer: 여러 개의 Attention 블록을 쌓아 Transformer 인코더 구성.
# CLIP: 이미지와 텍스트를 동시에 인코딩하고, 유사도를 계산하는 전체 모델.
# convert_weights: 모델 파라미터를 FP16으로 변환하여 효율성 향상.
# build_model: 저장된 상태 사전을 로드하여 CLIP 모델을 구축하고 초기화.
