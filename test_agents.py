import torch
import sys
import numpy as np
sys.path.append('phase4')
from agent import SAC, CQL, SVRL
import gymnasium as gym
from data.minari_loader import create_minari_dataloader

# 测试配置
base_config = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'gamma': 0.99,
    'tau': 0.005,
    'learning_rate': 3e-4,
    'hidden_size': 256
}

# 加载数据集
print('\n=== 加载数据集 ===')
data_loader, stats = create_minari_dataloader(
    dataset_name='D4RL/pen/expert-v2',
    batch_size=32,
    device=base_config['device']
)
print('数据加载器创建成功！')

# 获取数据维度
batch = next(iter(data_loader))
print('\n数据维度:')
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f'{key}: {value.shape}')

# 设置状态和动作维度
state_size = batch['observations'].shape[1]
action_size = batch['actions'].shape[1]
print(f'\n使用维度: state_size={state_size}, action_size={action_size}')

print('\n=== 测试SAC算法 ===')
sac = SAC(state_size, action_size, base_config)
print('SAC智能体创建成功！')

print('\n=== 测试CQL算法 ===')
cql_config = base_config.copy()
cql_config.update({
    'cql_weight': 1.0,
    'cql_tau': 10.0,
    'target_action_gap': 5.0,
    'num_random_actions': 10,
    'with_lagrange': True
})
cql = CQL(state_size, action_size, cql_config)
print('CQL智能体创建成功！')

print('\n=== 测试SVRL算法 ===')
svrl_config = base_config.copy()
svrl_config.update({
    'n_action_sample': 10,
    'mask_prob': 0.5,
    'lambda_struct': 0.5,
    'rank': 10
})
svrl = SVRL(state_size, action_size, svrl_config)
print('SVRL智能体创建成功！')

print('\n测试SAC更新...')
sac.set_normalization_stats(stats)
sac_metrics = sac.update(batch)
print('SAC指标:', sac_metrics)

print('\n测试CQL更新...')
cql.set_normalization_stats(stats)
cql_metrics = cql.update(batch)
print('CQL指标:', cql_metrics)

print('\n测试SVRL更新...')
svrl.set_normalization_stats(stats)
svrl_metrics = svrl.update(batch)
print('SVRL指标:', svrl_metrics)

# 测试动作选择
print('\n=== 测试动作选择 ===')
test_state = torch.randn(state_size).to(base_config['device'])
print(f'测试状态维度: {test_state.shape}')

sac_action = sac.get_action(test_state)
print('SAC动作形状:', sac_action.shape)

cql_action = cql.get_action(test_state)
print('CQL动作形状:', cql_action.shape)

svrl_action = svrl.get_action(test_state)
print('SVRL动作形状:', svrl_action.shape)

# 测试模型保存和加载
print('\n=== 测试模型保存和加载 ===')
sac.save('sac_test.pth')
cql.save('cql_test.pth')
svrl.save('svrl_test.pth')

sac.load('sac_test.pth')
cql.load('cql_test.pth')
svrl.load('svrl_test.pth')
print('所有模型保存和加载测试成功！')

print('\n所有测试完成！') 