"""
创建离线安装包
用于打包整个应用，方便在内网环境分发
"""

import shutil
import os
from pathlib import Path
import zipfile

def create_offline_package():
    """创建离线安装包"""
    
    print("="*60)
    print("创建拧紧曲线智能分析系统 - 离线安装包")
    print("="*60)
    
    # 基础目录
    base_dir = Path(__file__).parent
    package_dir = base_dir / "offline_distribution"
    
    # 清理旧包
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir()
    
    # 1. 复制离线依赖包
    print("\n[1/5] Copying offline packages...")
    offline_packages_src = base_dir / "streamlit_app" / "offline_packages"
    offline_packages_dst = package_dir / "offline_packages"
    if offline_packages_src.exists():
        shutil.copytree(offline_packages_src, offline_packages_dst)
        count = len(list(offline_packages_dst.glob('*.whl')))
        print(f"  [OK] Copied {count} packages")
    else:
        print("  [WARN] Offline packages not found")
    
    # 2. 复制应用代码
    print("\n[2/5] Copying application code...")
    app_dirs = [
        "streamlit_app",
        "src",
        "API",
        "output"
    ]
    for dir_name in app_dirs:
        src = base_dir / dir_name
        dst = package_dir / dir_name
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            print(f"  [OK] Copied {dir_name}/")
    
    # 3. 复制关键文件
    print("\n[3/5] Copying key files...")
    key_files = [
        "train.py",
        "dtw_cluster_full.py",
        "compare_models.py",
        "requirements.txt",
        "README.md",
        "MODEL_COMPARISON.md"
    ]
    for file_name in key_files:
        src = base_dir / file_name
        if src.exists():
            shutil.copy2(src, package_dir)
            print(f"  [OK] Copied {file_name}")
    
    # 4. Create installation scripts
    print("\n[4/5] Creating installation scripts...")
    
    # Windows installation script
    install_bat = package_dir / "install.bat"
    with open(install_bat, 'w', encoding='utf-8') as f:
        f.write('''@echo off
echo ==========================================
echo Tightening Curve Analysis - Offline Install
echo ==========================================
echo.
echo Installing dependencies (offline mode)...
echo This may take a few minutes...
echo.

cd offline_packages

echo [1/6] Installing base dependencies...
pip install --no-index --find-links=. numpy-2.4.3-*.whl

echo [2/6] Installing scientific computing...
pip install --no-index --find-links=. scipy-1.17.1-*.whl
pip install --no-index --find-links=. pandas-2.3.3-*.whl

echo [3/6] Installing machine learning...
pip install --no-index --find-links=. scikit_learn-1.8.0-*.whl

echo [4/6] Installing deep learning...
pip install --no-index --find-links=. torch-2.10.0-*.whl

echo [5/6] Installing visualization...
pip install --no-index --find-links=. plotly-6.6.0-*.whl

echo [6/6] Installing web framework...
pip install --no-index --find-links=. streamlit-1.55.0-*.whl

cd ..

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo Usage:
echo   1. Double-click run.bat to start
echo   2. Browser will open http://localhost:8501
echo.
pause
''')
    print(f"  [OK] Created install.bat")
    
    # Run script
    run_bat = package_dir / "run.bat"
    with open(run_bat, 'w', encoding='utf-8') as f:
        f.write('''@echo off
echo Starting Tightening Curve Analysis System...
echo.
cd streamlit_app
streamlit run app.py
echo.
pause
''')
    print(f"  [OK] Created run.bat")
    
    # Training script
    train_bat = package_dir / "train_new_data.bat"
    with open(train_bat, 'w', encoding='utf-8') as f:
        f.write('''@echo off
echo ==========================================
echo Train New Data (Offline Mode)
echo ==========================================
echo.
echo Steps:
echo   1. Put new data in API/Anord.json
echo   2. Press any key to start training
echo.
pause

echo.
echo [1/3] Running CNN training...
python train.py --model cnn --epochs 50 --save_model

echo.
echo [2/3] Running DTW clustering...
python dtw_cluster_full.py

echo.
echo [3/3] Training complete!
echo New model saved to output/
echo.
pause
''')
    print(f"  [OK] Created train_new_data.bat")
    
    # 5. Create README
    print("\n[5/5] Creating README...")
    readme = package_dir / "README.txt"
    with open(readme, 'w', encoding='utf-8') as f:
        f.write('''Tightening Curve Analysis System - Offline Version
================================================

System Requirements:
  - Windows 10/11 64-bit
  - Python 3.11+ (installed)
  - RAM: 4GB+
  - Disk: 2GB+

Installation:
  1. Extract this zip file
  2. Double-click install.bat
  3. Wait for installation (5-10 min)

Usage:
  1. Double-click run.bat
  2. Browser opens http://localhost:8501
  3. Use left sidebar to navigate

Train New Data:
  1. Prepare new data (JSON format)
  2. Replace API/Anord.json
  3. Double-click train_new_data.bat
  4. Wait for training

Features:
  Dashboard - System overview
  Data Explorer - Browse curves
  3D Visualization - Interactive 3D
  Cluster Analysis - DTW clustering
  Prediction - CNN quality check
  Cluster Comparison - Multi-dim view

Note:
  - Fully offline after installation
  - All training done locally
  - No data uploaded to servers

Version: v1.0.0
Date: 2026-03-12
''')
    print(f"  [OK] Created README.txt")
    
    # 6. Create ZIP package
    print("\n[6/6] Creating ZIP package...")
    zip_path = base_dir / "tightening-ai-offline.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
    
    package_size = zip_path.stat().st_size / (1024*1024)  # MB
    print(f"  [OK] Created tightening-ai-offline.zip ({package_size:.1f} MB)")
    
    print("\n" + "="*60)
    print("Offline package created!")
    print("="*60)
    print(f"\nOutput: {zip_path}")
    print(f"Size: {package_size:.1f} MB")
    print("\nDistribution:")
    print("  1. Copy zip to target machine")
    print("  2. Extract and run install.bat")
    print("  3. Run run.bat to start")
    print("="*60)


if __name__ == '__main__':
    create_offline_package()
