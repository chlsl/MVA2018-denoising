#!/usr/bin/env python3
# vim: set fileencoding=utf-8

"""
convert from .py to .ipynb (jupyter notebook) preserving blocks
Copyright (C) 2018, Gabriele Facciolo <gfacciol@gmail.com>
"""

# convert from .py to .ipynb

from __future__ import absolute_import, division, print_function

from nbformat.v3 import nbpy
import nbformat as nbf

from io import StringIO
import sys


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage:\n %s input.py output.ipynb'%sys.argv[0])
        sys.exit(1)


    n = nbf.v3.new_notebook()
    n['worksheets'].append( nbf.v3.new_worksheet())
    cells = n['worksheets'][0]['cells']
    #x = nbf.v3.new_text_cell('markdown')
    #x['source'] = 'bbb'
    #cells.append(x)

    buff=u''
    current_cell='code'
    c = 0

    # Read using v3
    with open(sys.argv[1],encoding='utf-8') as fin:
        def end_current(buff,current_cell):
            x = nbf.v3.new_text_cell(current_cell)
            buff=buff.lstrip()
            if current_cell=='markdown':
                x['source'] = buff.rstrip()
            else:
                x['input'] = buff.rstrip()
                x['outputs'] = ''
            if c==0 and buff=='': # skip if first cell is empty
                return
            cells.append(x)

        for l in fin.readlines():
            if '#%% Markdown [ ]:' in l:
                end_current(buff,current_cell)
                current_cell='markdown'
                #print(current_cell)
                c = c+1
                buff=''

            elif '#%% In [ ]:' in l:
                end_current(buff,current_cell)
                current_cell='code'
                #print(current_cell)
                c = c+1
                buff=''

            else: # strip leading '#' from markdown cells
                if current_cell=='markdown':
                    if l[0:2] == '# ':  # Formated by ipynb2py.py
                       buff = buff+'\n'+l[2:-1]
                    elif l[0] == '#':   #Strange format (without space)
                       buff = buff+'\n'+l[1:-1]
                    else:               # What is this, code?
                       buff = buff+'\n'+l
                       if l.lstrip() is not '':
                           print('This line of code was in the markdown block:\n{}'.format(l.encode('utf-8')))

                else:
                    buff = buff+l

        if buff!='':
            end_current(buff,current_cell)

        print (len(cells))

        # Write using the most recent version
        with open(sys.argv[2], 'w') as fout:
            nbf.write(n, fout) #, version=max(nbf.versions))
