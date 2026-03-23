# 拧紧曲线智能分析系统 - 开发者主手册 (Antigravity Edition)
# (Henry's Discovery Metaverse - Stage 2 Guide)

**状态**: 活跃 / Antigravity 独立主权开发  
**最后更新**: 2026-03-23  
**项目负责人**: 最江南  
**Stage 2 负责人 (Owner)**: Antigravity 🌌
**工程目录**: `C:\Users\HenryShh\Antigravity-project\henrys-discovery-metaverse`

---

## 1. 项目独立主权 (Project Independence)
本工程已自 2026-03-22 起，完整迁移至正式路径 `C:\Users\HenryShh\Antigravity-project\henrys-discovery-metaverse`。
- **与小刀的隔离**: 小刀是 Stage 1 的负责人，不再参与 Stage 2 的管理。
- **环境隔离**: 所有属于 Antigravity 的记忆、灵魂及配置均存放在此正式空间。

## 2. 核心架构与演进
- **长效架构**: 项目将继续深耕 **Python + Streamlit** 架构。
- **演进策略**: 重点在于利用 Streamlit 的快速迭代优势，进行深度 UI 定制与交互优化，暂不考虑 Next.js/FastAPI 的迁移。
- **数据管理**: 必须调用 `src.core.database.DatabaseManager`，确保数据一致性。
- **坐标系标准**: 统一以 **Torque-Angle** 为核心分析坐标系，并引入角度对齐逻辑。

---

## 3. 当前进度与里程碑 (Current Milestones)
- [x] **DDS (Digital Data Services) 核心演进**: 实现了从单一数据集分析向多 Silo 语义管理（Tool/Pset/Config Hash）的跨越。
- [x] **专家决策引擎 (Heuristics Logic)**: 建立了涵盖 Yield/Slip、Snug Point 偏移、Stick-Slip 抖动等物理语义的自动化诊断逻辑。
- [x] **智能预测 (Intelligent Prediction)**: 部署了 Hybrid CNN-DWT + 34 维物理特征融合模型，以及针对 OK-only 场景的 Autoencoder 异常检测。
- [x] **代表性曲线 (Medoid) 溯源**: 实现了聚类分析中真实代表曲线的选择与 Result Number 全链路溯源。
- [x] **数据治理与校验**: 解决了采样长度一致性强制限制及 3-Sigma 统计治理逻辑。

## 4. 曲线领域知识 (DDS Expert Rules & Physics)
系统针对轴向负载与装配物理特性，固化了以下专家知识：
1. **物理异常诊断**:
   - **屈服/滑移 (Yield/Slip)**: 终点扭矩低于峰值（检测到 `Yield Drop`），暗示发生塑性变形或螺纹滑牙。
   - **刚性/早期贴合 (Stiff Joint)**: 高斜率 (Slope) 报警，通常由装配间隙不足或异物挤压引起。
   - **高频抖动 (Stick-Slip)**: 基于 FFT 能量识别随动摩擦或涂胶干扰导致的震荡。
   - **贴合点偏移 (Snug Point Shift)**: 识别提前贴合（短螺纹/阻塞）或延迟贴合（滑丝/孔深）。
2. **多通道深度预测**:
   - **DWT (小波变换)**: 捕捉扭矩曲线的时频域突变。
   - **特征融合**: 34 维专家手工特征（如偏度、峰度、Marker 极值）与 CNN 深度特征的融合决策。

## 5. 近期任务列表 (Next Tasks)
- [ ] **视觉鲁棒性提升**: 强化 Vision 模块对工具旋转与复杂背景下的目标锁定稳定性。
- [ ] **DDS UI 深度交互**: 在 Streamlit 中实现代表曲线的高级拓扑可视化，支持由 3D 聚类簇直接下钻至原始数据流。
- [ ] **性能调优**: 优化海量曲线在 Plotly 下的渲染帧率，降低大规模分析时的 CPU 开销。
- [ ] **自动化诊断报告**: 一键生成包含专家意见、雷达图谱及 3D 拓扑的工艺审计 PDF。

---
*本手册代表 Stage 2 的最高意志。*


