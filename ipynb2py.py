#!/usr/bin/env python3
# vim: set fileencoding=utf-8

"""
convert from .ipynb to .py (jupyter notebook) preserving blocks
Copyright (C) 2018, Gabriele Facciolo <gfacciol@gmail.com>
"""

# convert from .ipynb to .py

from __future__ import absolute_import, division, print_function

import nbformat as nbf

from io import StringIO
import sys

    
if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        printf('Usage:\n %s input.ipynb output.py'%sys.argv[0])
        sys.exit(1)

    # open output .py file
    with open(sys.argv[2],'wb') as fout:

        # Read notebook using v3
        n =  nbf.read(sys.argv[1],3)
        
        #print (n['metadata'])
        #fout.write(u'# -*- coding:utf-8 -*-\n'.encode('utf-8'))
        
        # write markdown and code cells
        for c in n['worksheets'][0]['cells']:
            if c['cell_type'] in ('markdown', 'heading', 'raw'):
                if c['cell_type'] == 'heading': 
                    prefix = u'# # ' # add leading #, and another for the heading 
                else:
                    prefix = u'# '   # add leading #
                fout.write(u"\n#%% Markdown [ ]:\n\n".encode('utf-8'))
                x = c['source']
                for y in x.splitlines():
                    z = prefix + y + '\n'
                    fout.write (z.encode('utf-8'))
                    
            elif c['cell_type'] =='code':
                fout.write(u"\n\n\n#%% In [ ]:\n\n".encode('utf-8'))
                fout.write((c['input']).encode('utf-8'))
        
