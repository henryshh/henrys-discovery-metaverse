"""
Analyze cluster features based on actual data
"""

import json
import numpy as np
from pathlib import Path
import sys

# Fix encoding
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

def parse_anord_json(filepath):
    """Parse Anord.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    parts = content.split('}{')
    curves = []
    
    for i, part in enumerate(parts):
        if i == 0:
            json_str = part + '}'
        elif i == len(parts) - 1:
            json_str = '{' + part
        else:
            json_str = '{' + part + '}'
        
        try:
            obj = json.loads(json_str)
            if 'model' in obj and 'results' in obj['model']:
                model = obj['model']
                result = model['results'][0] if model['results'] else None
                if result and 'curves' in result:
                    points = result['curves'][0]['data']['points']
                    curves.append({
                        'id': i,
                        'resultNumber': model.get('resultNumber', i),
                        'report': model.get('report', 'UNKNOWN'),
                        'points': points
                    })
        except:
            pass
    
    return curves


def extract_curve_features(points):
    """Extract statistical features from curve"""
    if len(points) < 10:
        return None
    
    torque = np.array([p.get('torque', 0) for p in points])
    angle = np.array([p.get('angle', 0) for p in points])
    
    features = {
        'n_points': len(points),
        'torque_max': float(np.max(torque)),
        'torque_min': float(np.min(torque)),
        'torque_mean': float(np.mean(torque)),
        'torque_std': float(np.std(torque)),
        'angle_max': float(np.max(angle)),
        'angle_range': float(np.max(angle) - np.min(angle)),
        'torque_slope': float((torque[-1] - torque[0]) / (angle[-1] - angle[0] + 1e-8)),
        'torque_variance': float(np.var(torque)),
        'torque_diff_mean': float(np.mean(np.abs(np.diff(torque)))),
        'torque_diff_max': float(np.max(np.abs(np.diff(torque)))),
        'linearity': float(np.corrcoef(angle, torque)[0, 1]**2),
    }
    
    return features


def analyze_clusters():
    """Analyze each cluster's features"""
    # Load cluster results
    with open('output/dtw_full_results.json', 'r') as f:
        cluster_results = json.load(f)
    
    labels = np.array(cluster_results['labels'])
    n_clusters = cluster_results['n_clusters']
    
    # Load raw curve data
    curves = parse_anord_json('API/Anord.json')
    
    print("="*70)
    print("DTW Cluster Feature Analysis")
    print("="*70)
    print(f"Total curves: {len(curves)}")
    print(f"Clusters: {n_clusters}")
    print()
    
    cluster_analysis = []
    
    # Analyze each cluster
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(labels == cluster_id)[0]
        
        print(f"\n{'='*70}")
        print(f"Cluster {cluster_id}: {len(cluster_indices)} curves")
        print("="*70)
        
        cluster_features = []
        for idx in cluster_indices:
            if idx < len(curves):
                curve = curves[idx]
                feat = extract_curve_features(curve['points'])
                if feat:
                    cluster_features.append(feat)
        
        if not cluster_features:
            print("  No valid data")
            continue
        
        df = {k: [f[k] for f in cluster_features] for k in cluster_features[0].keys()}
        
        print(f"\n  Basic Statistics:")
        print(f"    Avg points: {np.mean(df['n_points']):.0f}")
        print(f"    Avg max torque: {np.mean(df['torque_max']):.3f} Nm")
        print(f"    Torque range: {np.min(df['torque_max']):.3f} - {np.max(df['torque_max']):.3f} Nm")
        print(f"    Avg angle range: {np.mean(df['angle_range']):.1f} degrees")
        
        print(f"\n  Shape Features:")
        print(f"    Avg slope: {np.mean(df['torque_slope']):.4f}")
        print(f"    Avg linearity (R2): {np.mean(df['linearity']):.3f}")
        print(f"    Avg fluctuation: {np.mean(df['torque_diff_mean']):.4f}")
        print(f"    Max fluctuation: {np.max(df['torque_diff_max']):.4f}")
        
        avg_slope = np.mean(df['torque_slope'])
        avg_linearity = np.mean(df['linearity'])
        avg_fluctuation = np.mean(df['torque_diff_mean'])
        
        print(f"\n  Feature Assessment:")
        if avg_linearity > 0.95:
            print(f"    [+] High linearity ({avg_linearity:.3f}) - near straight line")
        elif avg_linearity > 0.8:
            print(f"    [~] Medium linearity ({avg_linearity:.3f})")
        else:
            print(f"    [!] Low linearity ({avg_linearity:.3f}) - complex shape")
        
        if avg_fluctuation < 0.1:
            print(f"    [+] Low fluctuation ({avg_fluctuation:.4f}) - smooth curve")
        elif avg_fluctuation < 0.3:
            print(f"    [~] Medium fluctuation ({avg_fluctuation:.4f})")
        else:
            print(f"    [!] High fluctuation ({avg_fluctuation:.4f}) - has jitter")
        
        # OK/NOK distribution
        reports = [curves[idx]['report'] for idx in cluster_indices if idx < len(curves)]
        ok_count = sum(1 for r in reports if r == 'OK')
        nok_count = len(reports) - ok_count
        print(f"\n  Quality Distribution:")
        print(f"    OK: {ok_count} ({ok_count/len(reports)*100:.1f}%)")
        print(f"    NOK: {nok_count} ({nok_count/len(reports)*100:.1f}%)")
        
        # Store for summary
        cluster_analysis.append({
            'id': cluster_id,
            'count': len(cluster_indices),
            'linearity': avg_linearity,
            'fluctuation': avg_fluctuation,
            'ok_rate': ok_count / len(reports) * 100
        })
    
    # Summary
    print("\n" + "="*70)
    print("Summary - Cluster Characteristics")
    print("="*70)
    
    for ca in cluster_analysis:
        cid = ca['id']
        
        # Determine characteristics
        if ca['linearity'] > 0.95 and ca['fluctuation'] < 0.1:
            char = "Standard/Smooth"
        elif ca['fluctuation'] > 0.2:
            char = "High Fluctuation"
        elif ca['linearity'] < 0.8:
            char = "Non-linear"
        else:
            char = "Mixed"
        
        print(f"\nCluster {cid}: {char}")
        print(f"  Count: {ca['count']}, OK rate: {ca['ok_rate']:.1f}%")
        print(f"  Linearity: {ca['linearity']:.3f}, Fluctuation: {ca['fluctuation']:.4f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    analyze_clusters()
