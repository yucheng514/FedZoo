# CIFAR-10 对比实验命令集

本文档包含在CIFAR-10数据集上进行6个联邦学习算法对比实验的完整命令集。

## 数据集准备（首次运行）

```bash
cd /Users/huangyucheng/PycharmProjects/FedZoo
python dataset/generate_Cifar10.py noniid balance -
```

## 实验配置说明

- **数据集**: CIFAR-10 (10类图像分类)
- **客户端数**: 20
- **全局轮数**: 100
- **批大小**: 32 (CIFAR-10标准配置)
- **本地轮数**: 5
- **学习率**: 0.005 (基础) 或 0.01 (PerFedAvg)
- **评估间隔**: 1 (每轮评估)

## 6个算法的完整命令

### 1. FedAvg (基础联邦平均)
```bash
python main.py \
  -al FedAvg \
  -data CIFAR10 \
  -gr 100 \
  -nc 20 \
  -ncl 10 \
  -lbs 32 \
  -ls 5 \
  -lr 0.005 \
  -m CNN \
  --log_file logs/cifar10_fedavg.log
```

**参数说明**:
- `-al FedAvg`: 算法选择
- `-data CIFAR10`: 数据集
- `-gr 100`: 全局轮数
- `-nc 20`: 客户端总数
- `-ncl 10`: 类别数
- `-lbs 32`: 批大小
- `-ls 5`: 本地轮数
- `-lr 0.005`: 本地学习率
- `-m CNN`: 模型架构

---

### 2. FedAvg+FT (FedAvg+微调)
```bash
python main.py \
  -al FedAvg \
  -data CIFAR10 \
  -gr 100 \
  -nc 18 \
  -ncl 10 \
  -lbs 32 \
  -ls 5 \
  -lr 0.005 \
  -m CNN \
  -nnc 2 \
  --eval_new_clients \
  --fine_tuning_epoch_new 5 \
  --log_file logs/cifar10_fedavg_ft.log
```

**参数说明**:
- `-nnc 2`: 保留2个新客户端用于微调
- `--eval_new_clients`: 启用新客户端评估
- `--fine_tuning_epoch_new 5`: 微调轮数

---

### 3. Per-FedAvg (个性化FedAvg)
```bash
python main.py \
  -al PerFedAvg \
  -data CIFAR10 \
  -gr 100 \
  -nc 20 \
  -ncl 10 \
  -lbs 32 \
  -ls 5 \
  -lr 0.01 \
  -m CNN \
  --log_file logs/cifar10_perfedavg.log
```

**参数说明**:
- `-al PerFedAvg`: 个性化FedAvg算法
- `-lr 0.01`: Per-FedAvg推荐学习率（较高）
- 两步训练 + 一步微调机制自动运行

---

### 4. pFedMe (个性化FedMe)
```bash
python main.py \
  -al pFedMe \
  -data CIFAR10 \
  -gr 100 \
  -nc 20 \
  -ncl 10 \
  -lbs 32 \
  -ls 5 \
  -lr 0.005 \
  -m CNN \
  --beta 0.5 \
  --lamda 15.0 \
  --K 5 \
  --p_learning_rate 0.01 \
  --log_file logs/cifar10_pfedme.log
```

**参数说明**:
- `-al pFedMe`: pFedMe算法
- `--beta 0.5`: 全局/个性化参数插值因子 (0-1之间，0=完全个性化，1=完全全局)
- `--lamda 15.0`: Moreau包络正则化强度
- `--K 5`: 每批内的个性化优化步数
- `--p_learning_rate 0.01`: 个性化参数的学习率

---

### 5. IFCA (迭代联邦聚类)
```bash
python main.py \
  -al IFCA \
  -data CIFAR10 \
  -gr 100 \
  -nc 20 \
  -ncl 10 \
  -lbs 32 \
  -ls 5 \
  -lr 0.005 \
  --ifca_clusters 4 \
  --ifca_tau 5 \
  --ifca_mode clustered \
  --log_file logs/cifar10_ifca.log
```

**参数说明**:
- `-al IFCA`: IFCA聚类算法
- `--ifca_clusters 4`: 聚类簇数（推荐4个）
- `--ifca_tau 5`: IFCA本地轮数
- `--ifca_mode clustered`: 每轮重新分配簇（vs oneshot/local）

---

### 6. MCFL (多聚类联邦学习)
```bash
python main.py \
  -al MCFL \
  -data CIFAR10 \
  -gr 100 \
  -nc 20 \
  -ncl 10 \
  -lbs 32 \
  -ls 5 \
  -lr 0.005 \
  --mcfl_backbone auto \
  --mcfl_num_clusters 4 \
  --mcfl_seed 42 \
  --log_file logs/cifar10_mcfl.log
```

**参数说明**:
- `-al MCFL`: MCFL算法
- `--mcfl_backbone auto`: 自动选择CNN（CIFAR-10是图像）
- `--mcfl_num_clusters 4`: 聚类簇数
- `--mcfl_seed 42`: 随机种子，保证可复现性

---

## 快速执行脚本

如果想一次执行所有6个实验，可以创建脚本 `run_cifar10_experiments.sh`:

```bash
#!/bin/bash

cd /Users/huangyucheng/PycharmProjects/FedZoo

echo "=========== CIFAR-10 Comparison Experiments ==========="
echo ""

echo "[1/6] Running FedAvg..."
python main.py -al FedAvg -data CIFAR10 -gr 100 -nc 20 -ncl 10 -lbs 32 -ls 5 -lr 0.005 -m CNN --log_file logs/cifar10_fedavg.log

echo "[2/6] Running FedAvg+FT..."
python main.py -al FedAvg -data CIFAR10 -gr 100 -nc 18 -ncl 10 -lbs 32 -ls 5 -lr 0.005 -m CNN -nnc 2 --eval_new_clients --fine_tuning_epoch_new 5 --log_file logs/cifar10_fedavg_ft.log

echo "[3/6] Running Per-FedAvg..."
python main.py -al PerFedAvg -data CIFAR10 -gr 100 -nc 20 -ncl 10 -lbs 32 -ls 5 -lr 0.01 -m CNN --log_file logs/cifar10_perfedavg.log

echo "[4/6] Running pFedMe..."
python main.py -al pFedMe -data CIFAR10 -gr 100 -nc 20 -ncl 10 -lbs 32 -ls 5 -lr 0.005 -m CNN --beta 0.5 --lamda 15.0 --K 5 --p_learning_rate 0.01 --log_file logs/cifar10_pfedme.log

echo "[5/6] Running IFCA..."
python main.py -al IFCA -data CIFAR10 -gr 100 -nc 20 -ncl 10 -lbs 32 -ls 5 -lr 0.005 --ifca_clusters 4 --ifca_tau 5 --ifca_mode clustered --log_file logs/cifar10_ifca.log

echo "[6/6] Running MCFL..."
python main.py -al MCFL -data CIFAR10 -gr 100 -nc 20 -ncl 10 -lbs 32 -ls 5 -lr 0.005 --mcfl_backbone auto --mcfl_num_clusters 4 --mcfl_seed 42 --log_file logs/cifar10_mcfl.log

echo ""
echo "=========== All experiments completed ==========="
```

使用方法：
```bash
chmod +x run_cifar10_experiments.sh
./run_cifar10_experiments.sh
```

---

## 关键参数对比表

| 参数 | FedAvg | FedAvg+FT | Per-FedAvg | pFedMe | IFCA | MCFL |
|------|--------|-----------|-----------|--------|------|------|
| 模型数量 | 1(全局) | 1+2(微调) | 1+个性化 | 双轨 | K个集群 | K个集群 |
| 学习率 | 0.005 | 0.005 | 0.01 | 0.005 | 0.005 | 0.005 |
| 聚类数 | - | - | - | - | 4 | 4 |
| 微调 | 否 | 是(新客户端) | 一步(所有) | K步(所有) | 否 | 否 |
| 个性化方式 | 无 | 微调 | 两步+回退 | Moreau包络 | 聚类 | 元学习 |

---

## 结果日志位置

所有实验的日志将保存在 `logs/` 目录：
- `logs/cifar10_fedavg.log`
- `logs/cifar10_fedavg_ft.log`
- `logs/cifar10_perfedavg.log`
- `logs/cifar10_pfedme.log`
- `logs/cifar10_ifca.log`
- `logs/cifar10_mcfl.log`

## 资源估计

- **单个实验时间**: 1-4小时（GPU加速）
- **存储需求**: 约2GB（数据集+日志）
- **GPU内存**: 至少8GB推荐

## 常见问题

**Q: 我只想运行某个算法怎么办？**
A: 直接复制对应的命令并运行即可。

**Q: 如何改变全局轮数？**
A: 修改 `-gr` 参数，例如 `-gr 50` 表示50轮。

**Q: 如何改变客户端数量？**
A: 修改 `-nc` 参数，例如 `-nc 10` 表示10个客户端。

**Q: 如何比较不同种子下的结果？**
A: 目前可以多次运行同一命令，系统会自动使用不同的随机种子（通过MCFL/IFCA的seed���数）。


