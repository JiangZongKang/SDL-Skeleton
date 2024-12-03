import torch
import torch_pruning as tp
import os
import argparse
# from Ada_LSN.model import Network
# from Ada_LSN.genotypes import geno_inception as geno
from model import Network
from genotypes import geno_inception as geno

def analyze_model(model, prefix=""):
    """分析模型的结构和参数"""
    total_params = 0
    layer_info = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            layer_info.append({
                'name': name,
                'type': 'Conv2d',
                'in_channels': module.in_channels,
                'out_channels': module.out_channels,
                'kernel_size': module.kernel_size,
                'params': params
            })
            
    print(f"\n{prefix} Model Analysis:")
    print("-" * 80)
    print(f"Total parameters: {total_params:,}")
    print("\nLayer Details:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Type':<10} {'Shape':<20} {'Parameters':<15}")
    print("-" * 80)
    
    for layer in layer_info:
        shape = f"{layer['in_channels']}→{layer['out_channels']}"
        print(f"{layer['name']:<40} {layer['type']:<10} {shape:<20} {layer['params']:,}")
    
    return total_params, layer_info

def main():
    parser = argparse.ArgumentParser(description='Simple Magnitude Pruning for AdaLSN')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--model_path', type=str, 
                       default='/Users/jiangzongkang/Downloads/code/SDL-Skeleton/Ada_LSN/weights/skel_80000.pth')
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    print(f"\nLoading model from {args.model_path}")
    model = Network(128, 5, [0, 1, 2, 3], geno).to(device)
    model.load_state_dict(torch.load(args.model_path, 
                         map_location=lambda storage, loc: storage))
    model.eval()
    
    # 分析原始模型
    original_params, original_layers = analyze_model(model, "Original")
    
    # 准备剪枝
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    
    # 创建剪枝器
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')
    
    # 获取要忽略的层
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and m.out_channels == 1:
            ignored_layers.append(m)
        if hasattr(m, 'dsn') and isinstance(m, torch.nn.Conv2d):
            ignored_layers.append(m)

        # 建议增加的忽略条件
        # if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'cat'):
        #     ignored_layers.append(m)  # 特征融合层
        # if isinstance(m, torch.nn.BatchNorm2d):
        #     ignored_layers.append(m)  # 批归一化层
    
    # 配置剪枝器
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=args.pruning_ratio,
        ignored_layers=ignored_layers
    )

    # 记录初始模型信息
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Initial model params: {base_nparams/1e6:.2f}M, MACs: {base_macs/1e9:.2f}G")
    
    # 执行剪枝
    print(f"\nPruning with ratio {args.pruning_ratio}")
    pruner.step()

    # 统计当前模型信息
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Current model params: {nparams/1e6:.2f}M ({nparams/base_nparams*100:.1f}%)")
    print(f"Current model MACs: {macs/1e9:.2f}G ({macs/base_macs*100:.1f}%)")
    
    # 分析剪枝后的模型
    pruned_params, pruned_layers = analyze_model(model, "Pruned")
    
    # 计算变化
    reduction = (original_params - pruned_params) / original_params * 100
    print("\nPruning Results:")
    print("-" * 80)
    print(f"Original parameters: {original_params:,}")
    print(f"Pruned parameters:   {pruned_params:,}")
    print(f"Parameter reduction: {reduction:.2f}%")
    
    # 分析每层的变化
    print("\nLayer-wise Changes:")
    print("-" * 80)
    print(f"{'Layer Name':<40} {'Original':<15} {'Pruned':<15} {'Reduction %':<15}")
    print("-" * 80)
    
    for orig, pruned in zip(original_layers, pruned_layers):
        if orig['name'] == pruned['name']:
            reduction = (orig['params'] - pruned['params']) / orig['params'] * 100
            print(f"{orig['name']:<40} {orig['params']:<15,} {pruned['params']:<15,} {reduction:<15.2f}")
    
    # 保存剪枝后的模型
    save_path = os.path.join(os.path.dirname(args.model_path), 
                            f'pruned_model_mag_{args.pruning_ratio}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved pruned model to {save_path}")

if __name__ == '__main__':
    main() 