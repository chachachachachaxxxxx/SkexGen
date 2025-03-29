"""
Reconvert solids from OBJ format back to JSON format
"""
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from converter import OBJReconverter
import signal
from contextlib import contextmanager
import pickle

# Time out function
@contextmanager
def timeout(time):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)
    try:
        yield
    except TimeoutError:
        raise Exception("time out")
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

def raise_timeout(signum, frame):
    raise TimeoutError

NUM_THREADS = 10

def save_json_data(pathname, data):
    """Save data to a json file"""
    with open(pathname, 'w', encoding='utf8') as data_file:
        json.dump(data, data_file, indent=4)

def reconvert_folder_parallel(data):
    obj_file, output_folder = data
    save_file = Path(output_folder) / f"{obj_file.stem}.json"

    reconverter = OBJReconverter(obj_file)

    try:
        with timeout(30):
            json_data = reconverter.parse_obj()
            save_json_data(save_file, json_data)
    except Exception as ex:
        return [obj_file, str(ex)[:50]]
    return None

def find_files_already_processed_in_output_folder(output_folder):
    already_processed_files = set()
    for json_file in output_folder.glob("**/*.json"):
        already_processed_files.add(json_file.stem)
    return already_processed_files

def load_pkl_data(filepath):
    """Load data from a pickle file"""
    with open(filepath, 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing OBJ data")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to write the JSON output")
    parser.add_argument("--verbose", action="store_true", help="Print extra information about reconversion failures")
    args = parser.parse_args()

    output_folder = Path(args.output_folder)
    if not output_folder.exists():
        output_folder.mkdir()

    # Find the list of files which were already processed
    already_processed_files = find_files_already_processed_in_output_folder(output_folder)
    
    # Load OBJ data from pickle files
    obj_files = []
    output_folders = []

    train_data = load_pkl_data("/home/xxx/code/SkexGen/data/cad_data/train_deduplicate_se.pkl")
    invalid_data = load_pkl_data("/home/xxx/code/SkexGen/data/cad_data/train_invalid.pkl")
    train_data = [item for item in train_data if item['name'] not in invalid_data]

    val_data = load_pkl_data("/home/xxx/code/SkexGen/data/cad_data/val.pkl")
    test_data = load_pkl_data("/home/xxx/code/SkexGen/data/cad_data/test.pkl")

    all_data = train_data + val_data + test_data

    for obj_path in all_data:
        obj_file = Path(obj_path)
        if obj_file.stem not in already_processed_files:
            obj_files.append(obj_file)
            output_folders.append(output_folder / obj_file.parent.name)

    num_files_still_to_process = len(obj_files)
        
    assert len(output_folders) == num_files_still_to_process, "OBJ & JSON length mismatch"

    print(f"Found {len(obj_files)} files which require processing")
    
    if num_files_still_to_process > 0:
        # Parallel reconvert to JSON
        iter_data = zip(
            obj_files,
            output_folders,
        )
    
        reconvert_iter = Pool(NUM_THREADS).imap(reconvert_folder_parallel, iter_data) 
        for invalid in tqdm(reconvert_iter, total=len(obj_files)):
            if invalid is not None:
                if args.verbose:
                    print(f'Error reconverting {invalid}...')