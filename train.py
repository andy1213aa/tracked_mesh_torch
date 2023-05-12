from model import Extracter
from tfrecord.torch.dataset import TFRecordDataset
from readTFrecord import decode_image
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pytorch3d.loss import chamfer_distance
import einops
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 32
tfrecord_path = "../training_data/test.tfrecord"
mean_mesh_pth = '/home/aaron/Desktop/multiface/6674443_GHS/geom/vert_mean.bin'
KRT = "/home/aaron/Desktop/multiface/6674443_GHS/KRT"


def main():
    model = Extracter().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = get_dataloader()
    criterion = nn.MSELoss().to(device)
    # loss_chamfer, _ = chamfer_distance(vtx, pred_vtx)
    # loss = landmark_l2_loss().to(device)
    # criterion = customLoss().to(device)
    model_ft = train(model, dataloader, optimizer, criterion, EPOCHS)


def get_dataloader():
    index_path = None
    description = {
        "camID": "int",
        "img": "byte",
        "vtx": "byte",
        "texture": "byte",
        "verts_uvs": "byte",
        "faces_uvs": "byte",
        "verts_idx": "byte",
        "head_pose": "byte",
    }
    dataset = TFRecordDataset(
        tfrecord_path,
        index_path,
        description,
        transform=decode_image,
        # shuffle_queue_size=1024,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    return loader


class render_and_landmark():

    def __init__(self) -> None:
        super(render_and_landmark, self).__init__()

        from renderer import Renderer
        from face_landmark import FaceMesh

        self.renderer = Renderer((2048, 1334), (512, 334), device=device)
        self.face_landmark_detector = FaceMesh(batch_size=BATCH_SIZE,
                                               kpt_num=478)
        self._camera_info = KRT

        #init
        self._get_KRT()

    def _get_KRT(self):
        # 定義相機參數列表
        self.camera_params = {}
        with open(self._camera_info, 'r') as f:
            lines = f.readlines()
        i = 0

        while i < len(lines):
            # 讀取相機ID
            camera_id = int(lines[i].strip())
            i += 1
            # 讀取相機內部參數
            intrinsics = []
            for _ in range(3):
                intrinsics.append([float(x) for x in lines[i].strip().split()])
                i += 1
            intrinsics = np.array(intrinsics, dtype=np.float32).reshape((3, 3))

            #跳過一行
            i += 1

            # 讀取相機外部參數
            extrinsics = []
            for _ in range(3):
                extrinsics.append([float(x) for x in lines[i].strip().split()])
                i += 1
            extrinsics = np.array(extrinsics, dtype=np.float32).reshape((3, 4))
            # 添加相機參數到dict
            self.camera_params[camera_id] = {
                'K':
                torch.from_numpy(intrinsics),
                'RT':
                torch.from_numpy(extrinsics),
                'focal':
                torch.from_numpy(np.array([intrinsics[0, 0], intrinsics[1,
                                                                        1]])),
                'princpt':
                torch.from_numpy(np.array([intrinsics[0, 2], intrinsics[1,
                                                                        2]])),
            }

            #跳過一行
            i += 1

    def detect(
        self,
        pred_vtx,
        camID,
        img,
        texture,
        verts_uvs,
        faces_uvs,
        verts_idx,
        head_pose,
    ):

        pred_images = self.renderer.render(
            pred_vtx,
            texture,
            verts_uvs,
            faces_uvs,
            verts_idx,
            head_pose,
            torch.stack([self.camera_params[idx]['RT']
                         for idx in camID]).to(device),
            torch.stack([self.camera_params[idx]['focal']
                         for idx in camID]).to(device),
            torch.stack([self.camera_params[idx]['princpt']
                         for idx in camID]).to(device),
        )

        real_kpt = self.face_landmark_detector.detect(
            img.contiguous().cpu().numpy().astype('uint8'))
        
        pred_kpt = self.face_landmark_detector.detect(
            (255. * img).contiguous().cpu().numpy().astype('uint8'))

        real_kpt = torch.from_numpy(real_kpt).to(device).requires_grad_(True)
        pred_kpt = torch.from_numpy(pred_kpt).to(device).requires_grad_(True)

        return pred_images, real_kpt, pred_kpt
        # return torch.sqrt(torch.sum((real_kpt - pred_kpt)**2)) + loss_chamfer


class customLoss(nn.Module):

    def __init__(self):
        super(customLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.chamfer = chamfer_distance

    def forward(self, real_vtx, pred_vtx, real_kpts, fake_kpts):

        # vtx_loss, _ = self.chamfer(real_vtx, pred_vtx)
        kpt_loss = self.mse(real_kpts, fake_kpts)

        return kpt_loss


def train(net, dataloader, optimizer, criterion, epochs):
    net.train()
    bestLoss = 1e+10
    bestModel = net

    writer = SummaryWriter('runs')
    feature_detector = render_and_landmark()
    # read mean vertexes from .bin file.
    with open(mean_mesh_pth, 'rb') as f:
        data = f.read()
        mesh_mean = torch.from_numpy(
            np.frombuffer(data, dtype=np.float32).reshape(
                (7306, 3))).to(device).requires_grad_(False)
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):

            img = batch["img"].to(device)
            camID = batch['camID'].requires_grad_(False).numpy()
            vtx = batch['vtx'].to(device).requires_grad_(False)
            texture = batch['texture'].to(device).requires_grad_(False)
            verts_uvs = batch['verts_uvs'].to(device).requires_grad_(False)
            faces_uvs = batch['faces_uvs'].to(
                device, dtype=torch.int64).requires_grad_(False)
            verts_idx = batch['verts_idx'].to(
                device, dtype=torch.int64).requires_grad_(False)
            head_pose = batch['head_pose'].to(device).requires_grad_(False)

            shift_vtx = net(img)
            pred_vtx = shift_vtx + mesh_mean

            render_images, real_kpts, pred_kpts = feature_detector.detect(
                pred_vtx,
                camID,
                img,
                texture,
                verts_uvs,
                faces_uvs,
                verts_idx,
                head_pose,
            )

            loss = criterion(vtx, pred_vtx)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # 輸出即時結果

            print(
                f'Epoch: {epoch+1} Step: {(epoch+1)*(i+1)*BATCH_SIZE} L2 loss: {loss: 0.5f}'
            )

            # img = einops.rearrange(img, 'b h w c -> b c h w')
            # render_images = einops.rearrange(render_images,
            #                                  'b h w c -> b c h w')
            # writer.add_image(
            #     'input_images',
            #     img.to(torch.uint8),
            #     (epoch + 1) * (i + 1) * BATCH_SIZE,
            #     dataformats='NCHW',
            # )
            # writer.add_image(
            #     'pred_images',
            #     render_images,
            #     (epoch + 1) * (i + 1) * BATCH_SIZE,
            #     dataformats='NCHW',
            # )

            # writer.add_mesh('real_mesh', vtx)
            # writer.add_mesh('pred_mesh', pred_vtx)

        # if (epoch % 1 == 0):

        #     if loss < bestLoss:
        #         bestLoss = loss
        #         bestepoch = epoch
        #         bestModel = net

        # torch.save(
        #     net,
        #     "model/epoch" + str(bestepoch) + "_" + str(bestLoss) + ".pth")

    torch.save(bestModel, "model.pth")
    return net


if __name__ == '__main__':
    main()
