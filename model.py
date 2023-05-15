import torch.nn as nn

import einops


class Extracter(nn.Module):

    def __init__(self):
        super(Extracter, self).__init__()
        self.extracter = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 144, 3, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 144, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 216, 3, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(216, 216, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(216, 324, 3, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(324, 324, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(324, 486, 3, 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(486, 486, 3, 1),
            nn.ReLU(inplace=True),
        )
        
        self.pca = nn.Sequential(nn.Linear(4860, 160),
                                nn.Linear(160, 7306 * 3))

    def forward(self, img):
        x = einops.rearrange(img, 'b h w c -> b c h w')
        x = self.extracter(x)
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nn.Dropout(p=0.2)(x)
        x = self.pca(x)
        x = einops.rearrange(x, 'b (v c) -> b v c', c=3)
        return x

    # def forward(self, img, vtx_mean):
    #     x = einops.rearrange(img, 'b h w c -> b c h w')
    #     x = self.extracter(x)
    #     x = einops.rearrange(x, 'b h w c -> b (h w c)')
    #     x = nn.Dropout(p=0.2)(x)
    #     x = self.pca(x)
    #     x = einops.rearrange(x, 'b (v c) -> b v c', c=3)
    #     return x + vtx_mean