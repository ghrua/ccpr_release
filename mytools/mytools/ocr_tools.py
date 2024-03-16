import argparse
import pytesseract
from PIL import Image
from glob import glob
from mytools.tool_utils import FileUtils
import fire
from tqdm import tqdm


def call_tesseract(img_prefix, lang, save_path):
    fnames =  sorted(list(glob(img_prefix + "*.jpg")))
    data = []
    for img_path in tqdm(fnames):
        text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
        for line in text.split("\n"):
            if line:
                data.append(line)
    FileUtils.save_file(data, save_path, 'txt')


if __name__ == "__main__":
    fire.Fire({
        "call_tesseract": call_tesseract
    })