"""Read native memory estimates for an existing project without running or modifying it."""
import argparse
import json
from pathlib import Path
import numpy as np
from fdtd_pipeline.lumerical import load_lumapi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lumapi-path', default='D:/Program Files/Lumerical/v241/api/python/lumapi.py')
    args = parser.parse_args()
    api = load_lumapi(args.lumapi_path)
    f = api.FDTD(hide=True)
    try:
        f.load(str(args.project.resolve()))
        check = f.runsystemcheck()
        recommended_key = 'Memory_Recommended_Bytes' if 'Memory_Recommended_Bytes' in check else 'Memory_Recommended'
        selected = {key: check[key] for key in (recommended_key, 'Approximate_Memory_Requirements',
                                              'Total_FDTD_Yee_Nodes', 'Memory_Details')}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(selected, indent=2,
                              default=lambda value: np.asarray(value).tolist()), encoding='utf-8')
        print(json.dumps({key: selected[key] for key in (recommended_key, 'Total_FDTD_Yee_Nodes')},
                         default=lambda value: np.asarray(value).tolist()))
    finally:
        f.close()
