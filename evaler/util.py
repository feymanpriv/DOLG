# -*- coding: utf-8 -*-
"""utility functions"""
import sys, os
import numpy as np
import pickle


def l2_norm(fea):
    """l2 norm feature get cos dis"""
    if fea is None:
        return fea
    if len(fea.shape) > 1:
        fea = fea / np.linalg.norm(fea, axis=1)[:, None]
    else:
        fea = fea / np.linalg.norm(fea)
    return fea


def norm(x):
    n = np.sqrt(np.sum(x ** 2,1)).reshape(-1,1)
    return x / (n + 0.000001)


def walkfile(spath):
    """get files in input spath """
    files = os.listdir(spath)
    for file in files:
        tmppath = os.path.join(spath, file)
        if not os.path.isdir(tmppath):
            yield tmppath
        else:
            for lowfile in walkfile(tmppath):
                yield lowfile


def conmbineFeaPickle(COMBINE_DIR, OUTFILE):
    all_dict = {}
    files = os.listdir(COMBINE_DIR)
    for file in files:
        tmppath = os.path.join(COMBINE_DIR, file)
        print(tmppath)
        with open(tmppath,'rb') as fin:
            tmpres = pickle.load(fin)
            all_dict.update(tmpres)
    print(len(all_dict.keys()))
    with open(OUTFILE,'wb') as fout:
        pickle.dump(all_dict, fout, protocol=2)    
   

def visualize(path):
    searchfile = os.path.join(path, "searchresults.pickle")
    outhtml = os.path.join(path, "searchresults.html")
    with open(searchfile, "rb") as f:
        resdic = pickle.load(f)
    serverpath = 'http://127.0.0.0:8901/'
    print(serverpath)

    html_file = open(outhtml, "w")
    html_file.write('<html><meta charset=\"utf-8\"><body>\n')
    html_file.write('<p>\n')
    html_file.write('<table border="1">\n')

    for i, (k,v) in enumerate(sorted(resdic.items())):
        html_file.write('<td><img src="%s" width="150" height="150" /></td>' % \
                                (serverpath + "query/" + k))
        html_file.write("<td> %s %s</td>\n" % (i, k))

        recall = v[:10] # top10
        if len(recall) == 0:
            html_file.write('</tr>\n')
            continue
        for i in range(len(recall)):
            referhtml = serverpath + 'gallery/' + recall[i][0]
            html_file.write('<td><img src="%s" width="150" height="150" \
                /><br/><color="green">score: %s<br/> </td>' % (referhtml, str(recall[i][1])))
        html_file.write('</tr>\n')

    html_file.write("</table>\n</p>\n</body>\n</html>")
    print(path, "draw html finished ")



if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('tools.py command', file=sys.stderr)
