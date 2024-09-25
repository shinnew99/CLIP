import hashlib
import os
import urllib
import warnings
from packaging import version
from typing import Union, List

import torch
from PIL import Image
from torchvision.transformers import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .sample_tokenizer import SimpleTokenier as _Tokenizer

try:
    from torchvision.transformers import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if version.parse(torch.__version__) < version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1or higher is recommended")

    __all__ = ["available_models", "load", "tokenize"]
    _tokenizer = _Tokenizer()

    _MODELS = {
        "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    }

    # 모델 다운로드 함수 - 모델 파일을 지정된 URL에서 다운로드 하고, SHA256 해시를 검증하는 함수.
    # 근데 sha256은 어디에 쓰이지?
    def _download(url: str, root:str):
        os.makedirs(root, exist_ok = True) 
        filename = os.path.basename(url)

        expected_sha256 = url.split("/")[-2]  # URL의 2번째 마지막 부분을 sha256 해시로 간주한다. 
        download_target = os.path.join(root, filename)

        if os.path.exists(download_target) and not os.path.isfile(download_target):  # 저장할 파일
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        
        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

        # 파일 다운로드        
        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(total=int(source.info().get("Content-Length")), n_cols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        # 다운로드 후 무결성 검증
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")
        
        return download_target
    # 주의할점:
    # SHA256 해시 추출 방식: 위 함수는 URL의 두번째 마지막 붑분을 SHA256 해시로 간주하고 있다. 이는 URL구조에 따라 다르게 설정될 수 있으므로 실제 모델 URL과 일치해야 한다.


    def _convert_image_to_rgb(image):
        return image.convert("RGB")
    

    # 이미지 전처리 파이프라인 - 이미지데이터를 모델에 입력하기 전에 전처리하는 변환 파이프라인을 정의
    def _transform(n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    
    # 현재 사용할 수 있는 사전 학습된 CLIP 모델의 이름을 반환하는 함수이다.
    def available_models() -> List[str]:
        """ Returns the names of available CLIP models"""
        return list(_MODELS.keys())
    

    # 지정된 이름의 사전 학습된 CLIP 모델을 다운로드하고 로드하는 함수이다.
    def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_availabe() else "cpu", jit: bool=False, download_root: str = None):
        """ Load a CLIP model

        Parameters
        ----------
        name : str
            A model name listed by 'clip.availabe_models()', or the path toa model checkpoint containing the state_dict

        device: Union[str, torch.device]
            The device ti put the loaded model
        
        jit: bool
            Whether to laod the optimized JIT model or more hackable non-JIT model (default).

        download_root: str
            path to download the model files; by default, it uses "~/.cache/clip"

        
        Returns
        ----------
        model: torch.nn.Module
            The CLIP model

        preprocess: Callable[[PIL.Image], torch.Tensor]
            A torchvision transform that converts a PIL image into a tensor that the returned model cantake as its input
        """


        if name in _MODELS: 
            model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cahce/clip"))
        elif os.path.isfile(name):  # 해당 경로의 모델 파일을 사용한다
            model_path = name
        else:
            raise RuntimeError(f"Model {name} not found; available models={available_models()}")
        

        with open(model_path, 'rb') as opened_file:
            # 모델 파일을 바이너리 모드로 열고 있음
            try:  # JIT 모델 로드 시도
                # loading JIT archive
                model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
                # Just-In-Time으로 최적화된 모델을 로드하려 시도한다, 성공하면 model을 로드하고
                state_dict = None  
            except RuntimeError:  # JIT 모델 로드 실패
                # loading saved state dict
                if jit:
                    warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                    jit = False
                state_dict = torch.load(opened_file, map_location="cpu")  # 상태 사전을 로드
        
        # 비 JIT 모델 로드
        if not jit:  # jit이 False인 경우
            model = build_model(state_dict or model.state_dict()).to(device)  # 로드된 state_dict를 사용하여 CLIP 모델을 구축한다, model을 보통 state_dict로 넣나 보다
            if str(device) == "cpu":
                model.float()  # 디바이스가 CPU인 경우, 모델을 float()으로 변환한다, model.state_dict()로 저장하면 K와 V의 dict 형태로 저장되는데 이떄 V가 tensor값이기 때문에 전부 float로 변환됨
            return model, _transform(model.visual.input_resolution)  # 모델과 전처리 파이프라인을 반환한다.
        
        # patch the device names - JIT 모델의 디바이스 패치
        # 디바이스 노드 생성
        device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])  #  torch.jit.trace는 빈 텐서를 지정된 디바이스로 이동시키는 트레이싱, 추적한다, 결론적으로 내 코드가 돌아가는 환경이 CPU인지, GPU인지 추적하는 코드
        # torch.jit.trace: 주어진 함수의 실행을 추적하고, 그 함수가 실행되는 과정을 기록하여 그래프 형태로 저장
        # torch.one([]): 크기가 없는 텐서를 만듦, 이 텐서는 device 정보를 설정하는데 사용
        # => .to(torch.device(device))를 통해 텐서를 지정된 장치로 이동시킴, device='cuda'면 텐서가 GPU로, device='cpu'면 텐서가 CPU로 이동함
        device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)[-1]]  # device_node는 생성된 그래프에서 "Device"관련 노드를 추출한다.
        # prim::Constant는 JIT 컴파일 시 장치 정보와 관련된 상수를 나타낸다.
        # [-1]로 마지막 장치 정보에 대한 정보를 tensor에 저장
        # 즉, JIT 모델 내 장치 설정을 반영해서 다른 사람이 이 코드를 사용하더라도 의도된 장비

        # JIT는 코드를 작성할 떄부터 들어가 있는 코드 작성자의 배려로 느껴짐
        # "이 코드를 clone 해가서 쓸 당신들의 환경에서 편의를 좀 더 제공하고자 내가 설정을 미리 해줄게"

        # 노드 속성 가져오기 함수
        def _node_get(node: torch._C.Node, key:str):  # _node_get는 노드의 특정 속성을 가져오는 헬퍼 함수이다, 여기서 C가 는 C++의 API를 파이썬에서 사용할 수 있는 라이브러리(PyTorch) 형태로 가지고 옴
            """Gets attributes of a node which is polymorphic over return type.
            From https://github.com/pytorch/pytorch/pull/82628          
            """
            sel = node.kindOf(key)  # kindOf: (key)의 타입을 반환. 정수, 문자열, 장치 정보 등등
            return getattr(node, sel)(key)
        
        # 디바이스 패치 함수
        def patch_device(module):  # 모델의 그래프에서 "cuda"로 시작하는 디바이스 노드를 찾아 지정된 device_node로 속성을 복사한다.
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []
            
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):  # 노드를 연결하는 역할인걸까..?
                     if "value" in node.attributeNames() and str(_node_get(node, "value").startswith("cuda")):
                         node.copyAttributes(device_node)
                        # 어떤 device든 node의 형태로 model.dict()에 있는 value들을 가져오겠다.

        
        model.apply(patch_device)  # 모델 전체에 patch_device를 적용
        patch_device(model.encode_image)  # 모델에 image와 text를 encoding하는 걸까?
        patch_device(model.encode_text)

        # patch dtype to float32 on CPU
        if str(device) == "cpu":
            float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[]) 
            float_input = list(float_holder.graph.findNode("aten::to").inputs()[-1])
            float_node = float_input.node()

            def patch_float(module):
                try:
                    graphs = [module.graph] if hasattr(module, "graph") else []
                except RuntimeError:
                    graphs = []

                if hasattr(module, "forward1"):
                    graphs.append(module.forward1.graph)

                for graph in graph:
                    for node in graph.findAllNodes("aten::to"):
                        inputs = list(node.inputs())
                        for i in [1,2]:  # dtype can be the second or third arugment to atten::to()
                            if _node_get(inputs[i].node(), "value") == 5:
                                inputs[i].node().copyAttributes(float_node)    
        return model, _transform(model.input_resolution.item())
    

        def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool=False) -> Union[torch.IntTensor, otorch.LongTensor]:  # 토크나이즈 함수-텍스트 데이터를 토큰화하여 모델이 처리할수 있는 형태로 변환하는 함수이다.
            """
            Returns the tokenized representation of given input string(s)

            Parameters
            ----------------
            texts: Union[str, List[str]]
                An input string or a listof input strings totokenize           
            
            context_length: int
                The context length to use; all CLIPmodels use 77 as the context length
            
            truncate: bool
                Whether to truncate the textin case its encoding is longer than the context length

            Returns
            ---------------
            A two-dimensional tensor containing the resulting tokens, shape=[number of input strings, context_length].
            We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
            """
            if isinstance(texts, str):  # 텍스트 리스트로 변환
                texts = [texts]
            
            sot_token = _tokenizer.encoder["<|startoftext|>"]  
            eot_token = _tokenizer.encoder["<|endoftext|>"]
            all_token = [[sot_token]+_tokenizer.encode(text)+[eot_token] for text in texts]  # _tokenizer.encode(text) 텍스트를 토큰 리스트로 변환, <|startoftext|>와 <|endoftext|> 토큰을 각각 시퀀스의 시작과 끝에 추가함
            # 결과 텐서 초기화
            if version.parse(torch.__version__) <version.parse("1.8.0"):
                result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
            else:
                result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
            # PyTorch 버전에 따라 LongTensor 또는 IntTensor를사용하여 결과 텐서를 초기화한다.

            # 토큰 시퀀스 채우기 
            for i, tokens in enumerate(all_tokens):  
                if len(tokens) > context_length:  # 토큰 시퀀스의 길이가 context_length를 초과하면
                    if truncate:  # truncate=True일때 시퀀스를잘라내고 마지막 토큰을 <|endoftext|>로 설정
                        tokens = tokens[:context_length]
                        tokens[-1] = eot_token
                    else:
                        raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
                result[i, :len(tokens)] = torch.tensor(tokens) 
            return result  # 토큰화된 텐서를 변환한다
        # 텍스트 토크나이징: 텍스트를 토큰 리스트로 변호나하여 모델이 처리할 수 있는형태로 만듦
        # 시퀀스 길이 관리: 토큰 시퀀스가 모델의 context_length를 초과하지 않도록 관리
        # 데이터 타입호환성: PyTorch 버전에 따라 적절한 데이터 타입을 사용하여 호환성을 유지한다. 