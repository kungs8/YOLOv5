# -*- encoding: utf-8 -*-
'''
@Software:   PyCharm
@Project :   train
@Time    :   2021-06-10 00:39
@Author  :   yanpenggong
@Contact :   yanpenggong@163.com
@Version :   1.0
'''
import argparse
import os


def get_opt():
    paser = argparse.ArgumentParser()
    # 加载权重文件
    paser.add_argument("--weights", type=str, default="yolo5s.pt", help="initial weights path")
    # 模型配置文件，网络结构，使用修改好的yolov5m.yaml 文件
    paser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    # 数据集配置文件，数据集路径，类名等，使用数据集方面等cat.yaml文件
    paser.add_argument("--data", type=str, default="./data/coco128.yaml", help="dataset.yaml path")
    # 超参数文件
    paser.add_argument("--hyp", type=str, default="./data/coco128.yaml", help="超参数路径")
    # 训练总轮次，1个epoch等于使用训练集中的全部样本训练一次，值越大模型越精确，训练时间也越长。
    paser.add_argument("--epochs", type=int, default=300)
    # 批次大小，一次训练所选取的样本数(显卡差就调整小点)
    paser.add_argument("--batch_size", type=int, default=16, help="所有GPU的总批量大小")
    # 输入图片分辨率大小，nargs="+" 表示参数可设置一个或多个
    paser.add_argument("--img_size", nargs="+", type=int, default=[640, 640], help="[train, test] 图像尺寸")
    # 是否采用矩形训练，默认False， 开启后可显著的减少推理时间
    paser.add_argument("--rect", action="store_true", help="矩形训练")
    # 是否接着打断训练上次的结果接着训练
    paser.add_argument("--resume", nargs="?", const=True, default=False, help="恢复最近的训练")
    # 不保存模型，默认False
    paser.add_argument("--nosave", action="store_true", help="只保存最终检查点")
    # 不进行test，默认False
    paser.add_argument("--notest", action="store_true", help="只在最终epoch进行test")
    # 不自动调整anchor，默认False
    paser.add_argument("--noautoanchor", action="store_true", help="不自动调整anchor")
    # 是否进行超参数进化，默认False
    paser.add_argument("--evolve", action="store_true", help="进化超参数")
    # 谷歌云盘bucket，一般不会用到
    paser.add_argument("--bucket", type=str, default="", help="谷歌云盘")
    # 是否提前缓存图片到内存，以加快训练速度，默认False
    paser.add_argument("--cache_images", action="store_true", help="缓存图像以加快训练")
    # 选用加权图像进行训练
    paser.add_argument("--image_weights", action="store_true", help="使用加权图像选择进行训练")
    # 训练的设备，cpu:0 (表示一个gpu设备cuda:0); 0, 1, 2, 3(多个gpu设备), 值为空时，训练时默认使用计算机自带的显卡或cpu
    paser.add_argument("--device", default="", help="cuda device, 即 0 or 0,1,2,3 or cpu")
    # 是否进行多尺度训练，默认False
    paser.add_argument("--multi_scale", action="store_true", help="改变 img-size +/- 50%%")
    # 数据集是否只有一个类别，默认False
    paser.add_argument("--single_cls", action='store_true', help="将多类数据训练为单类")
    # 是否使用Adam优化器
    paser.add_argument("--adam", action="store_true", help="使用torch.optim.Adam()优化器")
    # 是否使用跨卡同步BN，在DDP模型使用
    paser.add_argument("--sync_bn", action="store_true", help="使用 SyncBatchNorm，仅仅适用于DDP模型")
    # gpu编号
    paser.add_argument("--local_rank", type=int, default=-1, help="DDP参数, 不要修改")
    # DataLoader 的最大 worker 数量
    paser.add_argument("--workers", type=int, default=8, help="DataLoader的最大worker数量")
    # 训练结果所存放的路径，默认为 runs/train
    paser.add_argument("--project", default="./runs/train", help="保存到project/name")
    # W&B 记录的图像数，最大为100
    paser.add_argument("--entity", default=None, help="W&B entity")
    # 训练结果所在文件夹的名称，默认为exp
    paser.add_argument("--name", default="exp", help="保存到project/name")
    # 若现有的 project/name 存在，则不进行递增
    paser.add_argument("--exit_ok", action="store_true", help="现有项目/名称确定，不增加")
    paser.add_argument("--quad", action="store_true", help="quad dataloader")
    paser.add_argument("--linear_lr", action="store_true", help="linear 学习率")
    paser.add_argument("--label_smoothing", type=float, default=0.0, help="标签平滑 epsilon")
    paser.add_argument("--upload_dataset", action="store_true", help="Upload dataset as W&B artifact table")
    paser.add_argument("--bbox_interval", type=int, default=-1, help="设置 W&B 的边界框图像记录间隔")
    paser.add_argument("--save_period", type=int, default=-1, help="在每个“save_period”时期后记录模型")
    # 记录最终训练的模型，即 last.pt
    paser.add_argument("--artifact_alias", type=str, default="latest", help="要使用的数据集工件的版本")
    opt = paser.parse_args()
    return opt


def main():
    # 参数设置
    opt = get_opt()
    print("opt:", opt)

    # 设置 DDP 变量
    opt.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    opt.global_rank = int(os.environ["RANK"]) if "RANK" in os.environ else -1
    set_logging(opt.global_rank)



if __name__ == '__main__':
    main()