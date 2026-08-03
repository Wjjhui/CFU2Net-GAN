import os
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from TU2Net import Generator_full
from Discriminator import Spatial, Temporal
from losses import Generator_loss_skillful, DiscriminatorLoss_hinge
from lr_scheduler import LambdaLinearScheduler
from utils import rainprint


# ================= Dataset =================
class MyDataset(Dataset):
    def __init__(self, data_path):
        super(MyDataset, self).__init__()
        self.files = os.listdir(data_path)
        self.root_path = data_path
        print(f"There are {len(self.files)} samples")

    def __getitem__(self, index):
        x = numpy.load(os.path.join(self.root_path, self.files[index]))
        x = torch.from_numpy(x)
        x=x*10
        return x[:4], x[4:10], x[10:14]

    def __len__(self):
        return len(self.files)


# ================= Train =================
def train():
    assert torch.cuda.is_available()
    device = "cuda"

    # ---------- paths ----------
    outpathtrain = r"./18cfu2netDmw"
    ckpt_dir = f"{outpathtrain}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = f"{ckpt_dir}/latest.pth"

    writer = SummaryWriter("runs/18dmw")

    # ---------- models ----------
    Generate_net = Generator_full(frames=6).to(device)
    Spatial_dis = Spatial().to(device)
    Temporal_dis = Temporal().to(device)

    # ---------- optimizers ----------
    Generate_net_optim = optim.Adam(Generate_net.parameters(), lr=2e-5, betas=(0.0, 0.999))
    Spatial_dis_optim = optim.Adam(Spatial_dis.parameters(), lr=2e-5, betas=(0.0, 0.999))
    Temporal_dis_optim = optim.Adam(Temporal_dis.parameters(), lr=2e-5, betas=(0.0, 0.999))

    # ---------- schedulers ----------
    cycle_lengths = [10000000]
    f_start, f_max, f_min = [0.01], [1.], [0.01]
    warm_up_steps = [1000]

    scheduler_Gen = lr_scheduler.LambdaLR(
        Generate_net_optim,
        lr_lambda=LambdaLinearScheduler(warm_up_steps, f_max, f_min, f_start, cycle_lengths).schedule)

    scheduler_Spa = lr_scheduler.LambdaLR(
        Spatial_dis_optim,
        lr_lambda=LambdaLinearScheduler(warm_up_steps, f_max, f_min, f_start, cycle_lengths).schedule)

    scheduler_Tem = lr_scheduler.LambdaLR(
        Temporal_dis_optim,
        lr_lambda=LambdaLinearScheduler(warm_up_steps, f_max, f_min, f_start, cycle_lengths).schedule)

    # ---------- losses ----------
    Generator_loss_f = Generator_loss_skillful().to(device)
    Discriminator_loss_f = DiscriminatorLoss_hinge().to(device)

    # ---------- data ----------
    dataset = MyDataset("G:\code\goesCodeW\goesDMW\merged_output2020")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)

    # ---------- resume ----------
    start_epoch = 0
    best_gen_loss = float("inf")

    if os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        Generate_net.load_state_dict(ckpt["gen"])
        Spatial_dis.load_state_dict(ckpt["spa"])
        Temporal_dis.load_state_dict(ckpt["tem"])

        Generate_net_optim.load_state_dict(ckpt["gen_optim"])
        Spatial_dis_optim.load_state_dict(ckpt["spa_optim"])
        Temporal_dis_optim.load_state_dict(ckpt["tem_optim"])

        scheduler_Gen.load_state_dict(ckpt["gen_sche"])
        scheduler_Spa.load_state_dict(ckpt["spa_sche"])
        scheduler_Tem.load_state_dict(ckpt["tem_sche"])

        start_epoch = ckpt["epoch"] + 1
        best_gen_loss = ckpt["best_gen_loss"]

        print(f"Resume from epoch {start_epoch}")

    # ---------- training ----------
    try:
        for epoch in range(start_epoch, 200):
            Generate_net.train()
            epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            gen_loss_sum = 0.0
            spa_loss_sum = 0.0
            tem_loss_sum = 0.0
            dis_loss_sum = 0.0

            for batch_idx, (x, y, dmw) in enumerate(epoch_pbar):
                x, y, dmw = x.float().to(device), y.float().to(device), dmw.float().to(device)

                # ----- Train Generator -----
                Generate_net_optim.zero_grad()
                print(x.shape,dmw.shape)
                gen_out = Generate_net(torch.squeeze(x), torch.squeeze(dmw))
                gen_out_copy = gen_out.detach().clone()

                spa_fake = Spatial_dis(gen_out)
                tem_fake = Temporal_dis(gen_out)
                dis_loss = (
                        Discriminator_loss_f(spa_fake, True)
                        + Discriminator_loss_f(tem_fake, True)
                )
                dis_loss_sum += dis_loss.item()

                Gen_loss = Generator_loss_f(y, gen_out, dis_loss)
                Gen_loss.backward()
                Generate_net_optim.step()

                gen_loss_sum += Gen_loss.item()

                # ----- Train Spatial D -----
                Spatial_dis_optim.zero_grad()
                spa_loss = (
                        Discriminator_loss_f(Spatial_dis(gen_out_copy), False)
                        + Discriminator_loss_f(Spatial_dis(y), True)
                )
                spa_loss.backward()
                Spatial_dis_optim.step()
                spa_loss_sum += spa_loss.item()

                # ----- Train Temporal D -----
                Temporal_dis_optim.zero_grad()
                tem_loss = (
                        Discriminator_loss_f(Temporal_dis(torch.cat([x, gen_out_copy], 1)), False)
                        + Discriminator_loss_f(Temporal_dis(torch.cat([x, y], 1)), True)
                )
                tem_loss.backward()
                Temporal_dis_optim.step()
                tem_loss_sum += tem_loss.item()

                scheduler_Gen.step()
                scheduler_Spa.step()
                scheduler_Tem.step()

                # 更新进度条显示
                if batch_idx % 10 == 0:
                    epoch_pbar.set_postfix({
                        'G_loss': f'{Gen_loss.item():.4f}',
                        'Spa_D': f'{spa_loss.item():.4f}',
                        'Tem_D': f'{tem_loss.item():.4f}',
                        'Dis': f'{dis_loss.item():.4f}'
                    })

            # 计算平均损失
            gen_loss_avg = gen_loss_sum / len(dataloader)
            spa_loss_avg = spa_loss_sum / len(dataloader)
            tem_loss_avg = tem_loss_sum / len(dataloader)
            dis_loss_avg = dis_loss_sum / len(dataloader)

            print(
                f"Epoch {epoch} | Gen Loss: {gen_loss_avg:.6f} | Spa Loss: {spa_loss_avg:.6f} | Tem Loss: {tem_loss_avg:.6f} | Dis Loss: {dis_loss_avg:.6f}")

            # ---------- TensorBoard记录 ----------
            # 记录主要损失
            writer.add_scalar("Loss/Generator_Total", gen_loss_avg, epoch)
            writer.add_scalar("Loss/Spatial_Discriminator", spa_loss_avg, epoch)
            writer.add_scalar("Loss/Temporal_Discriminator", tem_loss_avg, epoch)
            writer.add_scalar("Loss/Generator_Adversarial", dis_loss_avg, epoch)

            # 记录学习率
            writer.add_scalar("LR/Generator", scheduler_Gen.get_last_lr()[0], epoch)
            writer.add_scalar("LR/Spatial_D", scheduler_Spa.get_last_lr()[0], epoch)
            writer.add_scalar("LR/Temporal_D", scheduler_Tem.get_last_lr()[0], epoch)

            # 记录损失比率（帮助诊断GAN训练平衡）
            writer.add_scalar("Ratio/G_vs_Spa", gen_loss_avg / (spa_loss_avg + 1e-8), epoch)
            writer.add_scalar("Ratio/G_vs_Tem", gen_loss_avg / (tem_loss_avg + 1e-8), epoch)

            # ---------- save best ----------
            if gen_loss_avg < best_gen_loss:
                best_gen_loss = gen_loss_avg
                torch.save(Generate_net.state_dict(), f"{outpathtrain}/best_gen.pth")
                print(f"New best model! Loss: {best_gen_loss:.6f}")

            # ---------- save checkpoint ----------
            torch.save({
                "epoch": epoch,
                "best_gen_loss": best_gen_loss,
                "gen": Generate_net.state_dict(),
                "spa": Spatial_dis.state_dict(),
                "tem": Temporal_dis.state_dict(),
                "gen_optim": Generate_net_optim.state_dict(),
                "spa_optim": Spatial_dis_optim.state_dict(),
                "tem_optim": Temporal_dis_optim.state_dict(),
                "gen_sche": scheduler_Gen.state_dict(),
                "spa_sche": scheduler_Spa.state_dict(),
                "tem_sche": scheduler_Tem.state_dict(),
            }, ckpt_path)

            # 每10个epoch额外保存样本
            if epoch % 10 == 0:
                rainprint(torch.cat([x, gen_out_copy], 1),
                          f"{outpathtrain}/sample_epoch_{epoch}.jpg")
                torch.save(Generate_net.state_dict(), f"{outpathtrain}/Generate_pth/gen-{epoch}.pth")

                # 在TensorBoard中添加样本图像
                try:
                    # 选择前4个样本进行可视化
                    sample_to_show = torch.cat([x[:2], gen_out_copy[:2]], 1)
                    writer.add_images(f"Samples/epoch_{epoch}", sample_to_show, epoch)
                except Exception as e:
                    print(f"Warning: Could not save images to TensorBoard: {e}")

    except KeyboardInterrupt:
        print("Training interrupted, saving checkpoint...")
        torch.save({
            "epoch": epoch,
            "best_gen_loss": best_gen_loss,
            "gen": Generate_net.state_dict(),
            "spa": Spatial_dis.state_dict(),
            "tem": Temporal_dis.state_dict(),
            "gen_optim": Generate_net_optim.state_dict(),
            "spa_optim": Spatial_dis_optim.state_dict(),
            "tem_optim": Temporal_dis_optim.state_dict(),
            "gen_sche": scheduler_Gen.state_dict(),
            "spa_sche": scheduler_Spa.state_dict(),
            "tem_sche": scheduler_Tem.state_dict(),
        }, ckpt_path)
        print("Checkpoint saved safely.")

    finally:
        writer.close()  # 确保TensorBoard writer被正确关闭


if __name__ == "__main__":
    train()