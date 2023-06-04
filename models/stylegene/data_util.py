"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

face_class = ['background', 'head', 'head***cheek', 'head***chin', 'head***ear', 'head***ear***helix',
              'head***ear***lobule', 'head***eye***botton lid', 'head***eye***eyelashes', 'head***eye***iris',
              'head***eye***pupil', 'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
              'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair', 'head***hair***sideburns',
              'head***jaw', 'head***moustache', 'head***mouth***inferior lip', 'head***mouth***oral comisure',
              'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck', 'head***nose',
              'head***nose***ala of nose', 'head***nose***bridge', 'head***nose***nose tip', 'head***nose***nostril',
              'head***philtrum', 'head***temple', 'head***wrinkles']
face_must = ['head***forehead', 'head***frown', 'head***nose***bridge', 'head***nose', 'head***nose***ala of nose',
             'head***nose***nose tip', 'head***nose***nostril', 'head***mouth***inferior lip',
             'head***mouth***superior lip', 'head***chin', 'head***eye***top lid', 'head***eye***pupil',
             'head***eye***iris', 'head***eye***tear duct']

face_shape = ['head', 'head***chin', 'head***jaw', 'head***moustache', 'head***cheek']
