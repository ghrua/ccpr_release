import time
from tqdm.auto import tqdm as auto_tqdm
from tqdm import tqdm as manual_tqdm
import sys

tqdm = auto_tqdm if sys.stderr.isatty() else manual_tqdm

for i in tqdm(range(100)):
    time.sleep(0.1)
