import numpy as np
import pytest
import torch
from PIL import Image

import clip


@pytest.mark.parametrize('model_name', clip.available_models())
def test_consistency(model_name):  # 테스트함수, clip 모델을 2가지 방식으로 로드함
    device = "cpu"
    jit_model, transform = clip.load(model_name, device=device, jit=True)  # PyTorch의 JIT 컴파일러로 최적화된 모델
    py_model, _ = clip.load(model_name, device=device, jit=False)  # JIT 컴파일이 되지 않은 일반적인 PyTorch 모델

    image = transform(Image.open("CLIP.png")).unsqueeze(0).to(device)  # 테스트 이미지를 로드하고 CLIP 모델의 입력에 맞게 전처리한다, unsqueeze(0)는 차원을 추가하여 배치(batch) 처리를 위해 준비한다. 
    text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)   # 세 가지 텍스트("a diagram", "a dog", "a cat")를 CLIP의 tokenize 함수로 토큰화하여 모델에 입력할 수 있는 텐서로 변환한다.
    # image와 text를 다르게 처리하고 있는게보인다

    with torch.no_grad():  # 모델 추론 및 확률 계산, 두 모델(JIT 모델과 파이썬 모델)에 대해 각각 이미지와 텍스트를 입력하고, 그 결과로 나오는 로짓(logits)을 소프트맥스 함수로 확률로 변환한다.
        logits_per_image, = jit_model(image, text)
        jit_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        logits_per_image, _ = py_model(image, text)
        py_probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    assert np.allclose(jit_probs, py_probs, atol=0.01, rtol=0.1)  # 두 모델의 출력 확률(jit_probs, py_probs)이 서로 매우 유사한지 확인한다, np.allclose()는 두 값이 주어진 허용 오차(atol=0.01, rtol=0.1) 내에서 가까운지를 체크한다.

    # 이 테스트는 CLIP의 JIT 컴파일된 모델과 일반 모델이 동일한 입력에 대해 거의 동일한 결과를 내는지 확인하는 일관성 테스트이다.