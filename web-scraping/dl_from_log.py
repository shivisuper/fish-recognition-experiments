import requests
from progressbar import printProgressBar
from os import listdir, getcwd, remove
from os.path import join, isfile, splitext
from progressbar import printProgressBar

logpath = "/mnt/dissertation_work/logfiles"
logfiles = [file for file in listdir(logpath) if isfile(join(logpath, file))]
image_path = "/mnt/dissertation_work/image_urls/data_from_logs"

def iterate_over(file_name):
    with open(join(logpath, file_name), 'r') as handle:
        url_list = [line.rstrip() for line in handle]
        url_list = [url.split(' ')[0] for url in url_list if url.split(' ')[1] == '403']
        total_urls = len(url_list)
        counter = 0
        image_name = None
        printProgressBar(0, total_urls, prefix='Fetching images:',
                         suffix='Complete', length=50)
        for url in url_list:
            if 'png' in splitext(url)[1]:
                image_name = file_name.split('.')[0]+'_'+str(counter)+'.png'
            else:
                image_name = file_name.split('.')[0]+'_'+str(counter)+'.jpg'
            counter += 1
            download_image(url, image_name, counter, total_urls)
        print("All image scraped from {} file".format(file_name))

def download_image(url, file_name, counter, tot):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    try:
        req = requests.get(url, headers=headers)
        if req.status_code == requests.codes.ok:
            with open(join(image_path, file_name), 'wb') as f:
                for chunk in req:
                    f.write(chunk)
                printProgressBar(counter, tot, prefix='Fetching images:',
                                     suffix='Complete', length=50)
    except requests.exceptions.RequestException as re:
        pass


if __name__ == '__main__':
    for file in logfiles:
        iterate_over(file)
