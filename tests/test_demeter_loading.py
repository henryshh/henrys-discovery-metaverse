import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import TighteningDataLoader

def test_demeter_loading():
    data_path = 'API/demeter_result_2026-03-19-03-31-54.json'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    print(f"Testing loading from {data_path}")
    loader = TighteningDataLoader(data_path)
    loader.load()
    print(f"Loaded {len(loader.data)} records")
    
    loader.extract_curves(target_length=500)
    curves, labels, metadata = loader.get_data()
    
    print(f"Extracted {len(curves)} curves")
    if len(metadata) > 0:
        print(f"Sample metadata: {metadata[0]}")
        print(f"ResultNumber of first curve: {metadata[0].get('resultNumber')}")
        print(f"VIN of first curve: {metadata[0].get('vin')}")

if __name__ == "__main__":
    test_demeter_loading()
