import json

jsondata = {
    "word2vecfile": "/data2/xingbo/cscw2021/word2vec.bin",
    # "word2vecfile": "/data2/xingbo/cscw2021/GoogleNews-vectors-negative300.bin",
    "choidataset": "/data2/xingbo/cscw2021/text-segmentation/data/choi",
    "wikidataset": "/data2/xingbo/cscw2021/wiki_727",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
