from pathlib import Path

from nodes import NodePreprocess
from ways import WayPreprocess
from relation import RelationPreprocess
from segments import SegmentPreprocess
from dynamic import DynamicPreprocess


RAW_ROOT = Path("data/raw")
OSM_PATH = Path(RAW_ROOT / "osm_train_2019_01_03.json")


if __name__ == "__main__":
    node_process = NodePreprocess(
        RAW_ROOT, 
        OSM_PATH
    )
    node_process.preprocess()
    del node_process
    
    way_process = WayPreprocess(
        RAW_ROOT, 
        OSM_PATH
    )
    way_process.preprocess()
    del way_process

    segment_process = SegmentPreprocess(
        "data/raw", 
        OSM_PATH
    )
    segment_process.preprocess()
    del segment_process
    
    dynamic_process = DynamicPreprocess()
    dynamic_process.preprocess()
    del dynamic_process
    
    relation_process = RelationPreprocess()
    relation_process.preprocess()
    del relation_process
