from model import Extracter
from tfrecord.torch.dataset import TFRecordDataset
from readTFrecord import decode_image

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from pytorch3d.loss import chamfer_distance
import einops

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 32
tfrecord_path = "../training_data/test.tfrecord"


def main():
    model = Extracter().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dataloader = get_dataloader()
    loss = landmark_l2_loss().to(device)
    model_ft = train(model, dataloader, optimizer, loss, EPOCHS)


def get_dataloader():
    index_path = None
    description = {
        "img": "byte",
        "vtx": "byte",
        "vtx_mean": "byte",
        "tex": "byte",
        "verts_uvs": "byte",
        "faces_uvs": "byte",
        "verts_idx": "byte",
        "head_pose": "byte",
        "intricsic_camera": "byte",
        "extrinsic_camera": "byte"
    }
    dataset = TFRecordDataset(tfrecord_path,
                              index_path,
                              description,
                              transform=decode_image,
                              shuffle_queue_size=1024)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    return loader


class landmark_l2_loss(nn.Module):

    def __init__(self) -> None:
        super(landmark_l2_loss, self).__init__()

        from renderer import Renderer
        from face_landmark import FaceMesh

        self.renderer = Renderer((2048, 1334), (512, 334), device=device)
        self.face_landmark_detector = FaceMesh(batch_size=BATCH_SIZE,
                                               kpt_num=478)

    def forward(self, pred_vtx, img, vtx, tex, verts_uvs, faces_uvs, verts_idx,
                head_pose, focal, princpt, extrinsic_camera):

        pred_images = self.renderer.render(pred_vtx, faces_uvs, verts_uvs,
                                           verts_idx, tex, head_pose,
                                           extrinsic_camera, focal, princpt)

        pred_res, pred_kpt = self.face_landmark_detector.detect(pred_images)
        loss_chamfer, _ = chamfer_distance(vtx, pred_vtx)

        if not pred_res:
            return loss_chamfer

        real_res, real_kpt = self.face_landmark_detector.detect(img)
        real_kpt = torch.from_numpy(real_kpt).to(device)

        pred_kpt = torch.from_numpy(pred_kpt).to(device)
        return torch.sqrt(torch.sum((real_kpt - pred_kpt)**2)) + loss_chamfer


def train(net, dataloader, optimizer, criterion, epochs):
    net.train()
    bestLoss = 1e+10
    bestModel = net

    writer = SummaryWriter('runs')

    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):

            img = batch["img"].to(device)
            vtx = batch['vtx'].to(device)
            vtx_mean = batch['vtx_mean'].to(device)
            tex = batch['tex'].to(device)
            verts_uvs = batch['verts_uvs'].to(device)
            faces_uvs = batch['faces_uvs'].to(device, dtype=torch.int64)
            verts_idx = batch['verts_idx'][0].to(device, dtype=torch.int64)
            head_pose = batch['head_pose'].to(device)
            focal = batch['focal'].to(device)
            princpt = batch['princpt'].to(device)
            extrinsic_camera = batch['extrinsic_camera'].to(device)

            pred_vtx = net(img, vtx_mean)
            loss = criterion(pred_vtx, img, vtx, tex, verts_uvs, faces_uvs,
                             verts_idx, head_pose, focal, princpt,
                             extrinsic_camera)

            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            print(
                f'Epoch: {epoch+1} Step: {(epoch+1)*(i+1)*BATCH_SIZE} L2 loss: {loss: 0.5f}'
            )

        if (epoch % 1 == 0):
            pred_vtx = net(img, vtx_mean)
            render_images = criterion.renderer.render(
                pred_vtx,
                faces_uvs,
                verts_uvs,
                verts_idx,
                tex,
                head_pose,
                extrinsic_camera,
                focal,
                princpt,
            )

            img = einops.rearrange(
                img,
                'b h w c -> b c h w',
            )
            render_images = einops.rearrange(
                render_images,
                'b h w c -> b c h w',
            )

            writer.add_image('input_images',
                             img.to(torch.uint8),
                             (epoch + 1) * (i + 1) * BATCH_SIZE,
                             dataformats='NCHW')
            writer.add_image('pred_images',
                             render_images, (epoch + 1) * (i + 1) * BATCH_SIZE,
                             dataformats='NCHW')

            writer.add_mesh('real_mesh', vtx)
            writer.add_mesh('pred_mesh', pred_vtx)

            if loss < bestLoss:
                bestLoss = loss
                bestepoch = epoch
                bestModel = net

            torch.save(
                net,
                "model/epoch" + str(bestepoch) + "_" + str(bestLoss) + ".pth")

        torch.save(
            bestModel,
            "model/best_epoch" + str(bestepoch) + "_" + str(bestLoss) + ".pth")
    return net


if __name__ == '__main__':
    main()
