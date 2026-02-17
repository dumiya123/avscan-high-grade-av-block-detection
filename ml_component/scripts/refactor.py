
import shutil
import os
from pathlib import Path

def safe_move(src, dst):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        print(f"Skipping {src} (not found)")
        return
    
    # If destination is a directory, append filename
    if dst.is_dir():
        dst_file = dst / src.name
    else:
        dst_file = dst

    # If file exists, delete it first
    if dst_file.exists():
        print(f"Overwriting {dst_file}")
        if dst_file.is_dir():
            shutil.rmtree(dst_file)
        else:
            os.remove(dst_file)
    
    # Ensure parent dir exists
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    
    shutil.move(str(src), str(dst_file))
    print(f"Moved {src} -> {dst_file}")

base = Path("f:/Final_Year/Final_Semester_one/Final_Year_Research_Project/AtrionNet_Implementation/ml_component")
os.chdir(base)

# 1. Fix the file/folder confusion
# Check if src/engine is a file (it was losses.py)
if Path('src/engine').is_file():
    safe_move('src/engine', 'src/training/losses.py')

if Path('src/data_pipeline').is_file():
    safe_move('src/data_pipeline', 'src/data/loader.py')

if Path('src/modeling').is_file():
    safe_move('src/modeling', 'src/models/ecg_unet.py')

if Path('src/reporting').is_file():
    safe_move('src/reporting', 'src/reports/report_generator.py')


# 2. Create directories
dirs = ['src/data_pipeline', 'src/modeling', 'src/engine', 'src/reporting', 'src/utils']
for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)

# 3. Move contents
# Data
if Path('src/data').exists():
    for f in Path('src/data').glob('*'):
        if f.name == 'download.py':
            safe_move(f, 'scripts/download_data.py')
        else:
            safe_move(f, 'src/data_pipeline/')

# Models
if Path('src/models').exists():
    for f in Path('src/models').glob('*'):
        safe_move(f, 'src/modeling/')

# Training
if Path('src/training').exists():
    for f in Path('src/training').glob('*'):
        if f.name == 'train.py':
            safe_move(f, 'src/engine/trainer.py')
        elif f.name == 'evaluate.py':
            safe_move(f, 'src/engine/evaluator.py')
        else:
            safe_move(f, 'src/engine/')

# Reports
if Path('src/reports').exists():
    for f in Path('src/reports').glob('*'):
        if f.name == 'report_generator.py':
            safe_move(f, 'src/reporting/generator.py')
        else:
            safe_move(f, 'src/reporting/')

# Utils
if Path('src/utils.py').exists():
    print("Deleting old utils.py (will be split)")
    os.remove('src/utils.py')

# 4. Clean up empty dirs
for d in ['src/data', 'src/models', 'src/training', 'src/reports']:
    try:
        if Path(d).exists():
            Path(d).rmdir()
            print(f"Removed {d}")
    except:
        print(f"Could not remove {d} (not empty?)")

print("Refactoring Complete")
