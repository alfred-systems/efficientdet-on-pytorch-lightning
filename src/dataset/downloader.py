import os
import time
import json
import random
import shutil
from functools import reduce

import ray
import requests as rq
from datasets import load_dataset

ray.init()

@ray.remote
def download_urls(urls, output_dir):
    sucess_list = []
    for i, id, url in urls:
        output_path = os.path.join(output_dir, f'{i}.jpg')
        if os.path.exists(output_path):
            continue
        for _ in range(2):
            try:
                r = rq.get(url, stream=True)
                if r.status_code == 200:
                    with open(output_path, 'wb') as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)
                    sucess_list.append({'seq_id': i, 'sample_id': id})
                break
            except rq.exceptions.ConnectionError:
                time.sleep(random.random() * 2)
            if _ == 1:
                print(f'Fail to down load [{i}]{url}')
    return sucess_list
        

def download_laion4m_first_n(output_dir, n=20000, workers=8):
    dset = load_dataset("laion/laion400m")
    urls = [
        (i, dset['train'][i]['SAMPLE_ID'], dset['train'][i]['URL'])
        for i in range(n)
    ]
    os.makedirs(output_dir, exist_ok=True)
    splits = [urls[i::workers] for i in range(workers)]
    tasks = [download_urls.remote(split, output_dir) for split in splits]
    results = [ray.get(t) for t in tasks]
    
    sucess_list = reduce(lambda a, b: a + b, results)
    sucess_list = sorted(sucess_list, key=lambda x: x['seq_id'])
    with open(os.path.join(output_dir, "sucess_list.json"), mode='w') as f:
        json.dump(sucess_list, f)
    print('DONE')


download_laion4m_first_n('/home/ron_zhu/laion-400m/images', n=20000)