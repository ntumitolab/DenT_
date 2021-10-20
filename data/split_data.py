import os
import random
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='path to dataset', default='Spleen')
    parser.add_argument('--train_size', default=0.75, help='training set size')
    args = parser.parse_args()

    source = os.path.join(args.src, 'test')
    dest = os.path.join(args.src, 'val')

    #source_DNA = os.path.join(source, 'Source')
    source_Mito = os.path.join(source, 'Label')
    source_TL = os.path.join(source, 'Source')

    #dest_DNA = os.path.join(dest, 'DNA_Mask')
    dest_Mito = os.path.join(dest, 'Label')
    dest_TL = os.path.join(dest, 'Source')

    files = os.listdir(source_Mito)
    no_of_files = round(len(files) * (1 - args.train_size))

    for file_name in random.sample(files, 3):
        #shutil.move(os.path.join(source_DNA, file_name), dest_DNA)
        shutil.move(os.path.join(source_Mito, file_name), dest_Mito)
        shutil.move(os.path.join(source_TL, file_name), dest_TL)

if __name__ == '__main__':
    main()