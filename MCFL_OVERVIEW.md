# MCFL（Meta-Clustering Federated Learning）概述

本文档重点介绍 MCFL 的方法论和伪代码，便于纳入论文/项目 README 中。内容以中文撰写，分为：目标与动机、核心组件与流程、伪代码、关键超参数与实现细节、实验要点与复现命令、诊断/可视化建议。

---

## 1. 目标与动机

MCFL（Meta‑Clustering Federated Learning）旨在解决联邦学习中的两类现实挑战：

- 客户端异构性（non‑IID）：不同客户端的数据分布差异大；
- 时变分布（concept drift）：客户端数据分布随时间发生突变或周期性变化。

设计思想：把元学习（meta‑learning）的快速适应能力和基于客户端相似度的簇化聚合结合起来。通过双层更新（内层快速适配 + 外层簇级聚合）与动态聚类机制，MCFL 在分布突变时能快速在本地适配并依然保持高整体性能。

核心收益：
- raw_test_acc（不适应）用于揭示分布突变带来的直接性能退化；
- test_acc（含本地适应）展示了算法的鲁棒性（元学习带来的恢复能力）。

---

## 2. 核心组件

1) 双层优化结构
- 内层（inner‑loop / local adaptation）：每个客户端以簇模型为初始化，使用少量内步（inner steps）在本地快速适配；可视为对 MAML 风格 inner loop 的简化实现。
- 外层（federated/cluster aggregation）：服务器按簇聚合客户端的更新并更新簇模型。

2) 动态聚类
- 按客户端与簇模板的相似度计算 reassignment；
- 检测 outliers 并支持创建新簇；
- 单例簇（singleton）在条件满足时合并回现有簇（singleton merge），以控制簇数量波动。

3) Concept drift 注入（用于可控实验）
- 支持将某些客户端的数据在指定轮次进行样本或标签交换（`partner_map`），用于模拟 heavy drift、light drift 等情形。

4) 双指标评估
- raw_test_acc：直接用簇模型评估（adapt=False），反映模型在未适应时的脆弱性。
- test_acc：先做本地适配（adapt=True）再评估，反映元学习恢复性能。

---

## 3. 算法伪代码（可直接放入论文方法部分）

伪代码风格：尽量贴近实现，省略工程细节。

Server-side (MCFL):

```
Input: clients, T rounds, K_init clusters, inner_lr, inner_steps, local_epochs
initialize cluster_models for K_init clusters
initialize cluster assignments for clients

for t = 1 .. T:
    if drift_trigger(t):
        apply_partner_map_swap(clients, partner_map)    # 可选：heavy drift

    selected_clients = sample_clients(clients)
    client_results = []

    # 下发簇模型并让客户端做 inner-loop 适配与本地训练
    for client in selected_clients:
        model = cluster_models[client.cluster_id]
        adapted_model, meta_info = client_local_update(client, model,
                                                       inner_lr, inner_steps,
                                                       local_epochs)
        client_results.append((client.id, client.cluster_id, adapted_model, meta_info))

    # 簇级聚合（同簇内按样本数或权重平均）
    for cluster_id in clusters:
        updates = [r for r in client_results if r.cluster_id == cluster_id]
        if updates:
            cluster_models[cluster_id] = aggregate_updates(updates)

    # 动态聚类：计算相似度、重分配、创建/合并簇
    dynamic_clustering(clients, cluster_models)

    # 评估（两种模式）
    raw_test_accs = [c.evaluate(cluster_models[c.cluster_id], adapt=False) for c in clients]
    adapted_test_accs = [c.evaluate(cluster_models[c.cluster_id], adapt=True, inner_lr=inner_lr, local_epochs=inner_eval_epochs)
                         for c in clients]
    log_round(t, raw_test_accs, adapted_test_accs, clustering_events)
```

Client-side local update (简化)：

```
function client_local_update(client, model, inner_lr, inner_steps, local_epochs):
    temp_model = copy(model)
    # Inner-loop: fast adaptation on support set
    for k in 1..inner_steps:
        loss = compute_loss(temp_model, client.support_set)
        temp_model = temp_model - inner_lr * grad(loss)

    # Optional: local training epochs on full local data
    for e in 1..local_epochs:
        for batch in client.train_loader:
            loss = compute_loss(temp_model, batch)
            temp_model = temp_model - lr * grad(loss)

    return temp_model, {start_loss, end_loss}
```

注：在实现中，inner loop 可复用同一批 support/query 分割并返回 meta 信息（用于计算支持集/查询集损失），aggregate 可以是加权平均（权重按本地样本数）。

---

## 4. 关键超参数与实践建议

- mcfl_initial_clusters: 初始簇数（可与真实客户端数量/任务异质性匹配）；
- mcfl_enable_dynamic_clustering: 是否允许动态簇重分配（用于消融实验）；
- inner_lr 与 inner_steps: 控制本地快速适应的步长与步数；
- local_epochs: 客户端在 outer 轮中的本地训练强度（实验中通常设置为 1 或 5）；
- drift_type/drift_interval/drift_swap_clients: 用于合成 concept drift 的实验参数；
- outlier threshold / singleton merge 条件：控制簇数增长与合并行为（调参影响簇稳定性）。

实操建议：
- 若想观测 raw_test_acc 的明显 dip，先禁用 dynamic clustering（mcfl_enable_dynamic_clustering=0）并把 local_epochs 降到 1，以减少内层 + 聚类自适应的掩盖效应；
- 为可重复实验固定随机种子并把 num_workers 设为 0；
- 在 drift 触发处打印 partner_map 与每个被交换客户端的 label 分布摘要以验证 drift 是否生效。

---

## 5. 实验要点与复现命令

示例（基于项目默认实现，已用于生成 `logs/MCFL_20260511_221144.log`）：

基础复现命令（与日志里一致，注意给 `drift_swap_clients` 加引号以确保 argparse 正确解析）：

```bash
python main.py \
  --algorithm MCFL \
  --dataset CIFAR10 \
  --mcfl_initial_clusters 4 \
  --drift_type heavy \
  --drift_interval 30 \
  --drift_swap_clients "0-4,5-9" \
  --global_rounds 105 \
  --batch_size 32 \
  --mcfl_num_workers 0 \
  --local_epochs 5 \
  --seed 0 \
  --log_file logs/mcfl_drift_baseline.log
```

消融实验建议（用于展示方法学优势）：

- 关闭动态聚类（验证 dynamic clustering 是否掩盖 drift）
```bash
python main.py \ 
  --algorithm MCFL \ 
  --dataset CIFAR10 \ 
  --mcfl_initial_clusters 4 \ 
  --mcfl_enable_dynamic_clustering 0 \ 
  --drift_type heavy \ 
  --drift_interval 30 \ 
  --drift_swap_clients "0-4,5-9" \ 
  --global_rounds 105 \ 
  --batch_size 32 \ 
  --mcfl_num_workers 0 \ 
  --local_epochs 5 \ 
  --seed 0 \ 
  --log_file logs/mcfl_drift_nodyn.log
```

- 缩短本地适配（local_epochs=1）以放大适应带来的掩盖效应
```bash
python main.py \ 
  --algorithm MCFL \ 
  --dataset CIFAR10 \ 
  --mcfl_initial_clusters 4 \ 
  --drift_type heavy \ 
  --drift_interval 30 \ 
  --drift_swap_clients "0-4,5-9" \ 
  --global_rounds 105 \ 
  --batch_size 32 \ 
  --mcfl_num_workers 0 \ 
  --local_epochs 1 \ 
  --seed 0 \ 
  --log_file logs/mcfl_drift_le1.log
```

短轮次 debug（快速验证第 30 轮 drift 生效）：
```bash
python main.py \ 
  --algorithm MCFL \ 
  --dataset CIFAR10 \ 
  --mcfl_initial_clusters 4 \ 
  --mcfl_enable_dynamic_clustering 0 \ 
  --drift_type heavy \ 
  --drift_interval 30 \ 
  --drift_swap_clients "0-4,5-9" \ 
  --global_rounds 35 \ 
  --batch_size 32 \ 
  --mcfl_num_workers 0 \ 
  --local_epochs 1 \ 
  --drift_verbose 1 \ 
  --seed 0 \ 
  --log_file logs/mcfl_drift_debug.log
```

---

## 6. 诊断与可视化建议（便于结果上证据链）

- 绘图：raw_test_acc 与 test_acc 随轮次变化，图中标注 drift 触发轮（30/60/90）；
- 绘图：每轮 cluster_changes（reassigned 数量）、new_clusters、merged_singletons；
- 日志记录：在 drift 触发处打印 partner_map 与每个被交换 client 的样本数与标签频率（top‑k）；
- 计算并绘图：adapt_gain = test_acc - raw_test_acc（衡量元学习恢复能力）；
- 若要做理论对照：估计 path_length（簇最优点随时间的变化总和）与 aggregate cluster assignment error（误配率），用于解释跟踪误差。

---

## 7. 最后说明

该文档以可拷贝形式提供，适合放入实验方法部分或项目 README。若需要，我可以：

- 把 `drift_verbose` 的 debug 打印直接加到 `main.py`（仅在请求时开启）；
- 依据你的 target 论文格式（ICLR/NeurIPS/ICML）把本节改写成方法 + 算法框的 Latex/Markdown 版本。

文件已写入仓库：`MCFL_OVERVIEW.md`

欢迎告诉我是否需要我把其中部分内容（如伪代码）转成 LaTeX 环境或把示例图表脚本一并加入仓库。

