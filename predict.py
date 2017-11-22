import argparse
import json
import time
import datetime
import numpy as np
import code
import os
import cPickle as pickle
import math
import scipy.io

from imagernn.solver import Solver
from imagernn.imagernn_utils import decodeGenerator, eval_split
def main(params):
  checkpoint_path = params['checkpoint_path']
  print 'loading checkpoint %s' % (checkpoint_path, )
  checkpoint = pickle.load(open(checkpoint_path, 'rb'))
  checkpoint_params = checkpoint['params']
  dataset = checkpoint_params['dataset']
  model = checkpoint['model']
  misc = {}
  misc['wordtoix'] = checkpoint['wordtoix']
  ixtoword = checkpoint['ixtoword']

  blob = {} 
  blob['params'] = params
  blob['checkpoint_params'] = checkpoint_params
  blob['imgblobs'] = []

  root_path = params['root_path']
  img_names = open(os.path.join(root_path, 'tasks.txt'), 'r').read().splitlines()

  features_path = os.path.join(root_path, 'vgg_feats.mat')
  features_struct = scipy.io.loadmat(features_path)
  features = features_struct['feats'] 
  D,N = features.shape
  BatchGenerator = decodeGenerator(checkpoint_params)
  for n in xrange(N):
    print 'image %d/%d:' % (n, N)
    img = {}
    img['feat'] = features[:, n]
    img['local_file_path'] =img_names[n]

  
    kwparams = { 'beam_size' : params['beam_size'] }
    Ys = BatchGenerator.predict([{'image':img}], model, checkpoint_params, **kwparams)

   
    img_blob = {}
    img_blob['img_path'] = img['local_file_path']

   
    top_predictions = Ys[0]
    top_prediction = top_predictions[0] 
    candidate = ' '.join([ixtoword[ix] for ix in top_prediction[1] if ix > 0]) 
    print 'PRED: (%f) %s' % (top_prediction[0], candidate)
    img_blob['candidate'] = {'text': candidate, 'logprob': top_prediction[0]}    
    blob['imgblobs'].append(img_blob)

  
  save_file = os.path.join(root_path, 'result_struct.json')
  print 'writing predictions to %s...' % (save_file, )
  json.dump(blob, open(save_file, 'w'))
  html = ''
  for img in blob['imgblobs']:
    html += '<img src="%s" height="400"><br>' % (img['img_path'], )
    html += '(%f) %s <br><br>' % (img['candidate']['logprob'], img['candidate']['text'])
  html_file = os.path.join(root_path, 'result.html')
  print 'writing html result file to %s...' % (html_file, )
  open(html_file, 'w').write(html)

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_path', type=str, help='the input checkpoint')
  parser.add_argument('-r', '--root_path', default='example_images', type=str, help='folder with the images, tasks.txt file, and corresponding vgg_feats.mat file')
  parser.add_argument('-b', '--beam_size', type=int, default=1, help='beam size in inference. 1 indicates greedy per-word max procedure. Good value is approx 20 or so, and more = better.')

  args = parser.parse_args()
  params = vars(args) 
  print 'parsed parameters:'
  print json.dumps(params, indent = 2)
  main(params)
