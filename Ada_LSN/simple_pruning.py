import torch
import torch_pruning as tp
import os
import argparse
from model import Network
from genotypes import geno_inception as geno
from create_inception import BasicConv2d
import torch.nn as nn

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

def create_customized_pruner():
    """创建自定义的剪枝处理器"""
    
    class RelatedConvPruner(tp.pruner.function.BasePruningFunc):
        def prune_out_channels(self, layer, idxs):
            # 剪枝输出通道
            if not isinstance(idxs, list):
                idxs = idxs.tolist()
            keep_idxs = list(set(range(layer.out_channels)) - set(idxs))
            keep_idxs.sort()
            
            # 调整权重
            layer.out_channels = len(keep_idxs)
            layer.weight = torch.nn.Parameter(layer.weight[keep_idxs])
            if layer.bias is not None:
                layer.bias = torch.nn.Parameter(layer.bias[keep_idxs])
                
        def prune_in_channels(self, layer, idxs):
            # 剪枝输入通道
            if not isinstance(idxs, list):
                idxs = idxs.tolist()
            keep_idxs = list(set(range(layer.in_channels)) - set(idxs))
            keep_idxs.sort()
            
            # 调整权重
            layer.in_channels = len(keep_idxs)
            layer.weight = torch.nn.Parameter(layer.weight[:, keep_idxs])
            
        def get_out_channels(self, layer):
            return layer.out_channels
            
        def get_in_channels(self, layer):
            return layer.in_channels
    
    return {
        torch.nn.Conv2d: RelatedConvPruner()
    }

def update_bn_layer(bn, idxs):
    """更新BatchNorm层的参数"""
    keep_idxs = list(set(range(bn.num_features)) - set(idxs))
    keep_idxs.sort()
    
    bn.num_features = len(keep_idxs)
    bn.running_mean = bn.running_mean[keep_idxs]
    bn.running_var = bn.running_var[keep_idxs]
    if bn.affine:
        bn.weight = torch.nn.Parameter(bn.weight[keep_idxs])
        bn.bias = torch.nn.Parameter(bn.bias[keep_idxs])

class BasicConv2dPruner(tp.BasePruningFunc):
    def prune_out_channels(self, layer: nn.Module, idxs: list):
        # 剪枝conv层的输出通道
        tp.prune_conv_out_channels(layer.conv, idxs)
        # 同步更新bn层
        tp.prune_batchnorm_out_channels(layer.bn, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: list):
        # 剪枝conv层的输入通道
        tp.prune_conv_in_channels(layer.conv, idxs)
        return layer

    def get_out_channels(self, layer):
        return layer.conv.out_channels

    def get_in_channels(self, layer):
        return layer.conv.in_channels

def main():
    parser = argparse.ArgumentParser(description='Simple Magnitude Pruning for AdaLSN')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--model_path', type=str, 
                       default='/Users/jiangzongkang/Downloads/code/SDL-Skeleton/Ada_LSN/weights/skel_80000.pth')
    parser.add_argument('--pruning_ratio', type=float, default=0.5)
    args = parser.parse_args()
    
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
    example_inputs = (torch.randn(1, 3, 224, 224).to(device),)
    
    # 记录初始模型信息
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"Initial model params: {base_nparams/1e6:.2f}M, MACs: {base_macs/1e9:.2f}G")
    
    # 获取dsn层和对应的backbone层
    dsn_pairs = [
        ('dsn1', 'network.Conv2d_2b_3x3'),
        ('dsn2', 'network.Conv2d_4a_3x3'),
    ]
    
    # 创建层特定的剪枝比例字典
    pruning_ratio_dict = {}
    
    # 为dsn层和backbone层建立依赖关系
    named_modules = dict(model.named_modules())
    for dsn_name, backbone_name in dsn_pairs:
        dsn_module = getattr(model, dsn_name, None)
        backbone_module = named_modules.get(backbone_name)
        
        if dsn_module is not None and backbone_module is not None:
            print(f"Found DSN pair:")
            print(f"  DSN layer: {dsn_name}")
            print(f"  Backbone layer: {backbone_name}")
            print(f"  DSN in channels: {dsn_module.in_channels}")
            backbone_out_channels = backbone_module.conv.out_channels if hasattr(backbone_module, 'conv') else backbone_module.out_channels
            print(f"  Backbone out channels: {backbone_out_channels}")
            
            # 获取backbone的conv层
            conv_module = backbone_module.conv if hasattr(backbone_module, 'conv') else backbone_module
            
            # 将相关层添加到pruning_ratio_dict
            pruning_ratio_dict[conv_module] = args.pruning_ratio
            pruning_ratio_dict[dsn_module] = args.pruning_ratio
            
            # 如果有BatchNorm层，也添加到pruning_ratio_dict
            if hasattr(backbone_module, 'bn'):
                pruning_ratio_dict[backbone_module.bn] = args.pruning_ratio
    
    # 创建剪枝器
    imp = tp.importance.MagnitudeImportance(p=2, group_reduction='mean')
    
    # 配置剪枝器
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        pruning_ratio=args.pruning_ratio,
        customized_pruners={
            BasicConv2d: BasicConv2dPruner()
        },
        root_module_types=[BasicConv2d, torch.nn.Conv2d],
        global_pruning=False
    )
    
    # 执行剪枝
    for g in pruner.step(interactive=True):
        g.prune()
    
    print(model)
    
    # 验证通道数是否匹配
    def verify_channels():
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'dsn'):
                backbone_name = name.replace('dsn', 'network')
                backbone_module = dict(model.named_modules()).get(backbone_name)
                if backbone_module is not None:
                    print(f"Layer {name}:")
                    print(f"  DSN out channels: {m.out_channels}")
                    backbone_out_channels = backbone_module.conv.out_channels if hasattr(backbone_module, 'conv') else backbone_module.out_channels
                    print(f"  Backbone out channels: {backbone_out_channels}")
    
    print("\nVerifying channel consistency:")
    verify_channels()
    
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