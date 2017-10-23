import requests
from progressbar import printProgressBar
from os import listdir, getcwd, remove
from os.path import join, isfile, splitext

path_to_images = join(getcwd(), "dataset")

curr_dir = getcwd()

txtfiles = [f for f in listdir(curr_dir)
            if isfile(join(curr_dir, f)) and
            f.split('.')[-1] == "txt"]
try:
    txtfiles.remove('download_script.txt')
except ValueError as ve:
    print('download_script.txt not removed as it was not in list')

logfiles = [l for l in listdir(curr_dir)
            if isfile(join(curr_dir, l)) and
            l.split('.')[-1] == 'log']


def iterate_file(file_name):
    print("Opening file {}".format(file_name))
    with open(file_name, 'r') as handle:
        with open(join(curr_dir, file_name.split('.')[0] \
                       + '.log'), 'a') as logfile:
            counter = 0
            image_name = None
            file_list = [line.rstrip() for line in handle]
            l = len(file_list)
            printProgressBar(0, l, prefix='Fetching Images:',
                                 suffix='Complete', decimals=2, length=50)
            for line in file_list:
                if 'png' in splitext(line)[1].lower():
                    image_name = file_name.split('.')[0] + '_' + \
                            str(counter) + '.png'
                else:
                    image_name = file_name.split('.')[0] + '_' + \
                            str(counter) + '.jpg'
                counter += 1
                download_image(line, image_name, logfile, counter, l)
            print("All images scraped from {} file".format(file_name))


def download_image(url, file_name, logfile, counter, total):
    try:
        r = requests.get(url, stream=True)
        if r.status_code == requests.codes.ok:
            with open(join(path_to_images, file_name), 'wb') as f:
                for chunk in r:
                    f.write(chunk)
            printProgressBar(counter, total, prefix='Fetching Images:',
                             suffix='Complete', decimals=2, length=50)
        else:
            logfile.write("{} {}\n".format(url, r.status_code))
    except requests.exceptions.RequestException as e:
        logfile.write("{}\n".format(e))
    except Exception as ex:
        logfile.write("{}\n".format(ex))


if __name__ == '__main__':
    for log in logfiles:
        remove(join(curr_dir, log))
    for file_name in txtfiles:
        iterate_file(file_name)
