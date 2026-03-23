"""
工程管理使用示例
演示如何创建工程、导入数据、追加数据、保存模型
"""

from project_manager import ProjectManager


def demo():
    """演示工程管理功能"""
    
    # 初始化工程管理器
    pm = ProjectManager(workspace_dir="projects")
    
    print("="*60)
    print("[工具] 拧紧曲线工程管理演示")
    print("="*60)
    
    # ========== 1. 创建新工程 ==========
    print("\n[步骤1] 创建新工程")
    print("-"*60)
    
    try:
        config = pm.create_project(
            project_name="Anord生产线_2024Q1",
            description="Anord生产线第一季度拧紧数据",
            data_source="API/Anord.json"
        )
    except ValueError as e:
        print(f"工程已存在，加载现有工程: {e}")
        config = pm.load_project("Anord生产线_2024Q1")
    
    # ========== 2. 导入初始数据 ==========
    print("\n[步骤2] 导入初始数据")
    print("-"*60)
    
    # 假设有初始数据文件
    # pm.import_data(
    #     project_name="Anord生产线_2024Q1",
    #     data_file="API/Anord.json",
    #     data_format="json"
    # )
    
    print("[OK] 数据导入完成 (示例中已跳过)")
    
    # ========== 3. 追加新数据 ==========
    print("\n[步骤3] 追加同特征数据")
    print("-"*60)
    
    # 假设有新增数据
    # pm.append_data(
    #     project_name="Anord生产线_2024Q1",
    #     data_file="API/Anord_new.json",
    #     data_format="json"
    # )
    
    print("[OK] 数据追加完成 (示例中已跳过)")
    
    # ========== 4. 保存训练好的模型 ==========
    print("\n[步骤4] 保存训练模型")
    print("-"*60)
    
    # 假设模型已训练完成
    # pm.save_model(
    #     project_name="Anord生产线_2024Q1",
    #     model_path="output/cnn_model.pth",
    #     model_name="cnn_v1.pth",
    #     metrics={"accuracy": 0.98, "precision": 0.97, "recall": 0.99}
    # )
    
    print("[OK] 模型保存完成 (示例中已跳过)")
    
    # ========== 5. 查看工程信息 ==========
    print("\n[步骤5] 查看工程信息")
    print("-"*60)
    
    summary = pm.get_project_summary("Anord生产线_2024Q1")
    print(f"\n工程名称: {summary['name']}")
    print(f"工程ID: {summary['id']}")
    print(f"状态: {summary['status']}")
    print(f"数据: {summary['data_stats']['total_curves']} 条曲线")
    print(f"模型: {len(summary['models'])} 个")
    
    # ========== 6. 创建另一个工程（全新数据集） ==========
    print("\n[步骤6] 创建新工程（全新数据集）")
    print("-"*60)
    
    try:
        pm.create_project(
            project_name="新产线_测试数据",
            description="全新产线的测试数据",
            data_source="data/new_line.json"
        )
        print("[OK] 新工程创建成功")
    except ValueError as e:
        print(f"工程已存在: {e}")
    
    # ========== 7. 列出所有工程 ==========
    print("\n[步骤7] 列出所有工程")
    print("-"*60)
    
    projects = pm.list_projects()
    print(f"\n共有 {len(projects)} 个工程:\n")
    
    for p in projects:
        print(f"  [工程] {p['name']}")
        print(f"     状态: {p['status']} | 曲线: {p['curves']} | 模型: {p['models']}")
        print(f"     更新: {p['updated'][:19]}")
    
    print("\n" + "="*60)
    print("[OK] 演示完成！")
    print("="*60)
    print("\n使用说明:")
    print("  1. 命令行: python project_manager.py --help")
    print("  2. 创建工程: python project_manager.py create 工程名 --desc '描述'")
    print("  3. 导入数据: python project_manager.py import 工程名 数据文件.json")
    print("  4. 追加数据: python project_manager.py append 工程名 新数据.json")
    print("  5. 查看工程: python project_manager.py list")
    print("  6. 导出工程: python project_manager.py export 工程名 --output ./")
    print("="*60)


if __name__ == '__main__':
    demo()