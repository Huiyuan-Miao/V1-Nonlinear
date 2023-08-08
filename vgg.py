import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class vgg19(nn.Module):

    def __init__(self, num_classes = 1000,init_weights = False,dropout = 0.5):
        super(vgg19, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l11 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.l12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out',nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,0,0.01)
                    nn.init.constant_(m.bias,0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l11(x)
        x = self.relu(x)
        x = self.l12(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l21(x)
        x = self.relu(x)
        x = self.l22(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l31(x)
        x = self.relu(x)
        x = self.l32(x)
        x = self.relu(x)
        x = self.l33(x)
        x = self.relu(x)
        x = self.l34(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l41(x)
        x = self.relu(x)
        x = self.l42(x)
        x = self.relu(x)
        x = self.l43(x)
        x = self.relu(x)
        x = self.l44(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l51(x)
        x = self.relu(x)
        x = self.l52(x)
        x = self.relu(x)
        x = self.l53(x)
        x = self.relu(x)
        x = self.l54(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_outputs(self, x):
        module_list = []
        x = self.l11(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l12(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);
        module_list.append(np.array(x.detach().to('cpu')))

        x = self.l21(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l22(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);
        module_list.append(np.array(x.detach().to('cpu')))

        x = self.l31(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l32(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l33(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l34(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);
        module_list.append(np.array(x.detach().to('cpu')))

        x = self.l41(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l42(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l43(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.l44(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);
        module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l51(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l52(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l53(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l54(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))
        return module_list

class vgg19_modified2(nn.Module):

    def __init__(self, num_classes = 1000,init_weights = False,dropout = 0.5):
        super(vgg19_modified2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l11 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.l12 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.l21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out',nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,0,0.01)
                    nn.init.constant_(m.bias,0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l11(x)
        x = self.relu(x)
        x = self.l12(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l21(x)
        x = self.relu(x)
        x = self.l22(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l31(x)
        x = self.relu(x)
        x = self.l32(x)
        x = self.relu(x)
        x = self.l33(x)
        x = self.relu(x)
        x = self.l34(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l41(x)
        x = self.relu(x)
        x = self.l42(x)
        x = self.relu(x)
        x = self.l43(x)
        x = self.relu(x)
        x = self.l44(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l51(x)
        x = self.relu(x)
        x = self.l52(x)
        x = self.relu(x)
        x = self.l53(x)
        x = self.relu(x)
        x = self.l54(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def get_outputs(self,x):
        module_list = []
        x = self.l11(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l12(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l21(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l22(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l31(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l32(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l33(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l34(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l41(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l42(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l43(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l44(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l51(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l52(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l53(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l54(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))
        return module_list


class vgg19_modified3(nn.Module):

    def __init__(self, num_classes = 1000,init_weights = False,dropout = 0.5):
        super(vgg19_modified3, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.l11 = nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=1)
        self.l12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.l22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.l32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.l41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.l42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.l54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p = dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m,nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode = 'fan_out',nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias,0)
                elif isinstance(m,nn.Linear):
                    nn.init.normal_(m.weight,0,0.01)
                    nn.init.constant_(m.bias,0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l11(x)
        x = self.relu(x)
        x = self.l12(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l21(x)
        x = self.relu(x)
        x = self.l22(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l31(x)
        x = self.relu(x)
        x = self.l32(x)
        x = self.relu(x)
        x = self.l33(x)
        x = self.relu(x)
        x = self.l34(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l41(x)
        x = self.relu(x)
        x = self.l42(x)
        x = self.relu(x)
        x = self.l43(x)
        x = self.relu(x)
        x = self.l44(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.l51(x)
        x = self.relu(x)
        x = self.l52(x)
        x = self.relu(x)
        x = self.l53(x)
        x = self.relu(x)
        x = self.l54(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    def get_outputs(self,x):
        module_list = []
        x = self.l11(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l12(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l21(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l22(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l31(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l32(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l33(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l34(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l41(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l42(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l43(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l44(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))

        x = self.l51(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l52(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l53(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.l54(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.relu(x);module_list.append(np.array(x.detach().to('cpu')))
        x = self.maxpool(x);module_list.append(np.array(x.detach().to('cpu')))
        return module_list

