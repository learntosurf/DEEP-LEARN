# VGGNet

[*Very Deep Convolutional Networks for Large-Scale Image Recognition (2014)*](https://arxiv.org/abs/1409.1556)

- AlexNet이 등장하며 이미지 분류 분야에서 CNN 모델이 주목받기 시작했으며, 2014년 VGGNet과 GoogleNet은 ILSVRC 대회에서 더 깊은 레이어를 쌓아 AlexNet보다 높은 성능을 보여줌
- 깊은 신경망이 성능 향상에 미치는 영향을 실험적으로 입증한 초기 모델


### VGGNet Architecture
![VGGNet Architecture](https://github.com/user-attachments/assets/fa92564c-22bf-4df3-862c-01f17c51b639)

- **Input**: 224x224 RGB image
- **Component**:
  - **Convolution layers**: 
    - 3x3 filter, stride=1, padding=True
    - 채널 수 증가 (64 → 128 → 256 → 512)
  - **Pooling layers**:
    - 2x2 filter, stride=2 
    - 공간적 크기 축소 (224 → 112 → 56 → 28 → 14 → 7)
  - **Fully Connected layers**:
    - 4096 → 4096 → 1000
  - **Soft-max layer**: ImageNet 클래스 개수 (1000)

### Configurations

| **Configuration** | **Layers**           | **특징**                                    |
|-------------------|---------------------|-----------------------------------------|
| **A**            | 8 conv. + 3 FC (11) |                                           |
| **A-LRN**        | 8 conv. + 3 FC (11) | LRN (Local Response Normalization) 사용   |
| **B**            | 10 conv. + 3 FC (13)| conv3 2개 추가                            |
| **C**            | 13 conv. + 3 FC (16)| conv1 3개 추가                            |
| **D → VGG16**    | 13 conv. + 3 FC (16)| C의 conv1 → conv3으로 수정                |
| **E → VGG19**    | 16 conv. + 3 FC (19)| conv3 3개 추가                            |

- VGGNet은 작은 filter (3x3)를 다층으로 쌓아 동일한 Receptive field를 구현:
  - [3x3 filter로 3번 conv] = [5x5 filter로 2번 conv] = [7x7 filter로 1번]
  - 비선형성 증가, 학습 파라미터 감소 (49→50→27)


### VGG16 Dimension Changes
![Filter Size Comparison](https://github.com/user-attachments/assets/c75421a4-8f12-41a0-bc69-7d9f4511c338)

#### Input
- **Input**: 224x224x3

#### Convolutional Blocks

- **Block 1**
  - Conv1-1: 224x224x3 → 224x224x64
  - Conv1-2: 224x224x64 → 224x224x64
  - Maxpool1: 224x224x64 → 112x112x64

- **Block 2**
  - Conv2-1: 112x112x64 → 112x112x128
  - Conv2-2: 112x112x128 → 112x112x128
  - Maxpool2: 112x112x128 → 56x56x128

- **Block 3**
  - Conv3-1: 56x56x128 → 56x56x256
  - Conv3-2: 56x56x256 → 56x56x256
  - Maxpool3: 56x56x256 → 28x28x256

- **Block 4**
  - Conv4-1: 28x28x256 → 28x28x512
  - Conv4-2: 28x28x512 → 28x28x512
  - Maxpool4: 28x28x512 → 14x14x512

- **Block 5**
  - Conv5-1: 14x14x512 → 14x14x512
  - Conv5-2: 14x14x512 → 14x14x512
  - Maxpool5: 14x14x512 → 7x7x512

#### Fully Connected Layers
- Flatten: 7x7x512 → 25088
- FC1: 25088 → 4096
- FC2: 4096 → 4096
- Output: 4096 → 1000



## PyTorch Implementation

```python
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
```
- Flatten
    ```python
    x = x.view(x.size(0), -1)
    ```
    - 신경망의 첫번째 차원은 항상 batch size를 나타냄 (`x.size(0)`는 현재 입력된 tensor의 batch size를 가져옴)
    - 나머지 차원은 `-1`로 설정하여 자동으로 계산되도록 함
    - 다른 방법
        ```python
        x = torch.flatten(x, start_dim=1)
        ```       
- Conv layer가 2개 있는 block과 3개 있는 block을 따로 선언한 경우 
[https://velog.io/@euisuk-chung/파이토치-파이토치로-CNN-모델을-구현해보자-VGGNet편](https://velog.io/@euisuk-chung/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A1%9C-CNN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90-VGGNet%ED%8E%B8)
