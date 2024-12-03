import torch
import torch_pruning as tp
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from datasets.sklarge_RN import TrainDataset
from Ada_LSN.model import Network
from Ada_LSN.genotypes import geno_inception as geno
from engines.trainer_AdaLSN import Trainer, logging

class ModelPruner:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        
    def get_importance_criterion(self):
        """获取重要性评估准则"""
        criterion_dict = {
            'magnitude': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'l1norm': tp.importance.L1NormImportance(p=1, group_reduction='mean'),
            'random': None,  # RandomPruner doesn't need importance criterion
            'bnscale': tp.importance.BNScaleImportance(p=2, group_reduction='mean'),
            'groupnorm': tp.importance.GroupNormImportance(p=2, group_reduction='mean')
        }
        return criterion_dict.get(self.args.pruner_type)
        
    def get_ignored_layers(self):
        """获取不需要剪枝的层"""
        ignored_layers = []
        for m in self.model.modules():
            # 忽略最后的分类器层
            if isinstance(m, torch.nn.Conv2d) and m.out_channels == 1:
                ignored_layers.append(m)
            # 忽略dsn层
            if hasattr(m, 'dsn') and isinstance(m, torch.nn.Conv2d):
                ignored_layers.append(m)
        return ignored_layers
    
    def get_pruner(self, example_inputs):
        """获取指定的剪枝器"""
        importance = self.get_importance_criterion()
        ignored_layers = self.get_ignored_layers()
        
        pruner_dict = {
            'magnitude': tp.pruner.MagnitudePruner(
                self.model,
                example_inputs=example_inputs,
                importance=importance,
                iterative_steps=self.args.iterative_steps,
                pruning_ratio=self.args.pruning_ratio,
                global_pruning=self.args.global_pruning,
                ignored_layers=ignored_layers,
            ),
            'l1norm': tp.pruner.L1NormPruner(
                self.model,
                example_inputs=example_inputs,
                importance=importance,
                iterative_steps=self.args.iterative_steps,
                pruning_ratio=self.args.pruning_ratio,
                global_pruning=self.args.global_pruning,
                ignored_layers=ignored_layers,
            ),
            'random': tp.pruner.RandomPruner(
                self.model,
                example_inputs=example_inputs,
                iterative_steps=self.args.iterative_steps,
                pruning_ratio=self.args.pruning_ratio,
                global_pruning=self.args.global_pruning,
                ignored_layers=ignored_layers,
            ),
            'bnscale': tp.pruner.BNScalePruner(
                self.model,
                example_inputs=example_inputs,
                importance=importance,
                iterative_steps=self.args.iterative_steps,
                pruning_ratio=self.args.pruning_ratio,
                global_pruning=self.args.global_pruning,
                ignored_layers=ignored_layers,
            ),
            'groupnorm': tp.pruner.GroupNormPruner(
                self.model,
                example_inputs=example_inputs,
                importance=importance,
                iterative_steps=self.args.iterative_steps,
                pruning_ratio=self.args.pruning_ratio,
                global_pruning=self.args.global_pruning,
                ignored_layers=ignored_layers,
            ),
        }
        return pruner_dict.get(self.args.pruner_type, pruner_dict['magnitude'])
        
    def prune_model(self, example_inputs, dataloader):
        """执行迭代式模型剪枝"""
        print(f"开始使用 {self.args.pruner_type} 剪枝器进行剪枝")
        print(f"目标剪枝率: {self.args.pruning_ratio}, 迭代次数: {self.args.iterative_steps}")
        
        # 获取指定的剪枝器
        pruner = self.get_pruner(example_inputs)
        
        # 记录初始模型信息
        base_macs, base_nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
        print(f"Initial model params: {base_nparams/1e6:.2f}M, MACs: {base_macs/1e9:.2f}G")
        
        # 迭代式剪枝
        for i in range(self.args.iterative_steps):
            print(f"\nIteration {i+1}/{self.args.iterative_steps}")
            
            # 执行剪枝
            pruner.step()
            
            # 统计当前模型信息
            macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)
            print(f"Current model params: {nparams/1e6:.2f}M ({nparams/base_nparams*100:.1f}%)")
            print(f"Current model MACs: {macs/1e9:.2f}G ({macs/base_macs*100:.1f}%)")
            
            # 如果需要，在每次迭代后进行微调
            if self.args.finetune_after_each_iter:
                print("Finetuning after iteration...")
                self.model, ft_loss = finetune_model(self.model, dataloader, self.args)
                print(f"Finetuning loss: {ft_loss:.4f}")
        
        return self.model

def finetune_model(model, dataloader, args):
    """对剪枝后的模型进行微调"""
    print("开始微调...")
    
    # 配置优化器
    lr = args.ft_lr / args.iter_size
    optimizer = optim.Adam(model.parameter(args.ft_lr), lr=lr, 
                          betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    # 使用trainer进行微调
    trainer = Trainer(model, optimizer, dataloader, args)
    loss = trainer.train()
    return model, loss

def main():
    parser = argparse.ArgumentParser(description='Prune and Finetune AdaLSN')
    # 基本参数
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--data1', default='./SKLARGE/', type=str)
    parser.add_argument('--data2', default='./SKLARGE/train_pairRN60_255_s_all.lst', type=str)
    
    # 剪枝参数
    parser.add_argument('--pruner_type', type=str, default='magnitude',
                      choices=['magnitude', 'l1norm', 'random', 'bnscale', 'groupnorm'])
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    parser.add_argument('--iterative_steps', type=int, default=5)
    parser.add_argument('--global_pruning', action='store_true',
                      help='是否使用全局剪枝策略')
    parser.add_argument('--finetune_after_each_iter', action='store_true',
                      help='是否在每次迭代后进行微调')
    parser.add_argument('--model_path', type=str, 
                      default='./Ada_LSN/weights/inception_sklarge/skel_74000.pth')
    
    # 微调参数
    parser.add_argument('--ft_lr', type=float, default=1e-7)
    parser.add_argument('--ft_epochs', type=int, default=10)
    parser.add_argument('--iter_size', default=10, type=int)
    parser.add_argument('--weight_decay', default=0.0002, type=float)
    parser.add_argument('--max_step', default=10000, type=int)
    parser.add_argument('--disp_interval', default=100, type=int)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--resume_iter', default=0, type=int)
    parser.add_argument('--lr_step', default=70000, type=int)
    parser.add_argument('--lr_gamma', default=0.1, type=float)
    
    args = parser.parse_args()
    
    # 设置GPU
    torch.cuda.set_device(args.gpu_id)
    
    # 加载预训练模型
    print(f"Loading pretrained model from {args.model_path}")
    model = Network(128, 5, [0, 1, 2, 3], geno).cuda(args.gpu_id).eval()
    model.load_state_dict(torch.load(args.model_path, 
                         map_location=lambda storage, loc: storage))
    
    # 准备数据
    dataset = TrainDataset(args.data2, args.data1)
    dataloader = DataLoader(dataset, shuffle=True)
    
    # 执行剪枝
    pruner = ModelPruner(model, args)
    example_inputs = (torch.randn(1, 3, 224, 224).cuda(args.gpu_id),)
    pruned_model = pruner.prune_model(example_inputs, dataloader)
    
    # 如果没有在每次迭代后微调，那么在最后进行一次完整的微调
    if not args.finetune_after_each_iter:
        print("\nPerforming final finetuning...")
        finetuned_model, ft_loss = finetune_model(pruned_model, dataloader, args)
        print(f"Final finetuning loss: {ft_loss:.4f}")
    else:
        finetuned_model = pruned_model
    
    # 保存最终模型
    save_path = os.path.join(os.path.dirname(args.model_path), 
                            f'pruned_model_{args.pruner_type}_{args.pruning_ratio}.pth')
    torch.save(finetuned_model.state_dict(), save_path)
    print(f"Saved final model to {save_path}")

if __name__ == '__main__':
    main() 