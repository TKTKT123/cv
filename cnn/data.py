import numpy as np
from torchvision import transforms,datasets

tr="train"
te="test"
file_path=r"/gemini/data-1"

#定义transforms
transforms = transforms.Compose(
[

transforms.RandomResizedCrop(150),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
]

)

train_data = datasets.ImageFolder(os.path.join(file_path,tr), transforms)
test_data=datasets.ImageFolder(os.path.join(file_path,te), transforms)