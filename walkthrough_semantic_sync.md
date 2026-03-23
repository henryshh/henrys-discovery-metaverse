# Walkthrough: High-Resolution & Semantic Data Management

This walkthrough demonstrates the new capabilities for robust, context-aware curve analysis.

## 1. Memory-Efficient Streaming Loader
The [TighteningDataLoader](file:///c:/Users/HenryShh/Antigravity-project/henrys-discovery-metaverse/src/data_loader.py#12-216) was upgraded to a generator-based streaming model. This allows processing of files >200MB without proportional memory usage.

**Key File**: [src/data_loader.py](file:///c:/Users/HenryShh/Antigravity-project/henrys-discovery-metaverse/src/data_loader.py) (Methods: [load_stream](file:///c:/Users/HenryShh/Antigravity-project/henrys-discovery-metaverse/src/data_loader.py#22-39), [_stream_v1_concatenated](file:///c:/Users/HenryShh/Antigravity-project/henrys-discovery-metaverse/src/data_loader.py#40-68))

## 2. Semantic Hierarchy & Siloing
Incoming curves are no longer treated as a single flat list. They are automatically partitioned into "Silos" based on:
- **Tool Serial**: The specific tightening gun used.
- **Pset Number**: The specific program set ID.
- **Config Hash**: A fingerprint of the torque targets and tolerances (Pset micro-tweaks).

**Key File**: [src/services/dataset_service.py](file:///c:/Users/HenryShh/Antigravity-project/henrys-discovery-metaverse/src/services/dataset_service.py) (Method: [import_from_json](file:///c:/Users/HenryShh/Antigravity-project/henrys-discovery-metaverse/src/services/dataset_service.py#183-229))

## 3. High-Resolution Torque-Angle Alignment (Standard View)
The system now prioritizes **Torque vs. Angle** as the primary coordinate for quality analysis, following industry standards.

**Key Improvements**:
- **Standard View Overlay**: A new 2D plot in the Visualization tab defaults to Angle (°) on the X-axis, allowing direct comparison of curve profiles regardless of sampling rate.
- **Angle-based Clustering Analytics**: The cluster averaging and comparison logic now uses **Angular Interpolation**. This ensures that even if curves have different point densities (e.g., 500 vs 2000 points), their "Average Profile" is calculated correctly across the rotation range.
- **Auxiliary Coordinates**: Time (s) and Current (A) remain available in the **3D Trajectory Analysis** for debugging motor behavior and timing issues.

## 4. Automated Consistency Check (Sync Report)
When syncing new records to a silo, the system performs a statistical sanity check:
- **Parameter Check**: If targets change, a new silo variant is created automatically.
- **Drift Detection**: New records are compared against the silo's baseline using a **3-Sigma (Z-Score)** analysis of Final Torque.
- **Length Integrity Check**: Since data resolution (e.g., 500 vs. 2000 points) is hardware-defined, the system now strictly validates that new curves match the existing silo's point count. Any mismatch is flagged as a critical anomaly.

## 5. UI Integration
The import interface now provides a **Sync Status Report** showing exactly which silos were updated and alerting to any detected inconsistencies.

## Verification Status
- [x] Stream parsing of large JSON.
- [x] Auto-partitioning of multiple Psets from one file.
- [x] Statistical anomaly flagging on import.
- [x] Time-based plotting in Trajectory and Cluster Analysis.
