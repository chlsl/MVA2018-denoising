"""
* tools for displaying images in the notebook

Copyright (C) 2017-2018, Gabriele Facciolo <facciolo@cmla.ens-cachan.fr>
"""

from __future__ import print_function


### DISPLAY IMAGES AND TABLES IN THE NOTEBOOK


# utility function for printing with Markdown format
def printmd(string):
    from IPython.display import Markdown, display
    display(Markdown(string))

    
def printbf(obj):
    printmd("__"+str(obj)+"__")

    
def show_array(a, fmt='jpeg'):
    ''' 
    display a numpy array as an image
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)  
    '''
    try:
        import Image
    except ImportError:
        from PIL import Image
    from io import BytesIO
    import IPython.display
    import numpy as np
    f = BytesIO()
    PIL.Image.fromarray(np.uint8(a).squeeze() ).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))

    

def display_image(img):
    '''
    display_image(img)
    display an image in the curren IPython notebook
    img can be an url, a local path, or a numpy array
    '''
    from IPython.display import display, Image
    from urllib import parse   
    import numpy as np
    
    if type(img) == np.ndarray:
        x = np.squeeze(img).copy()
        show_array(x)
    elif parse.urlparse(img).scheme in ('http', 'https', 'ftp'):
        display(Image(url=img)) 
    else:
        display(Image(filename=img)) 
        
      
    
def display_imshow(im, range=None, cmap='gray', axis='equal', invert=False):
    '''
    display_imshow(img)
    display an numpy array using matplotlib
    img can be an url, a local path, or a numpy array
    range is a list [vmin, vmax]
    cmap sets the colormap ('gray', 'jet', ...) 
    axis sets the scale of the axis ('auto', 'equal', 'off')
          https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.axis.html
    invert reverses the y-axis
    '''
    import matplotlib.pyplot as plt  
    vmin,vmax=None,None
    if range:
        vmin,vmax = range[0],range[1]
    plt.figure(figsize=(13, 10))
    plt.imshow(im.squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
    if invert:
        plt.gca().invert_yaxis()
    plt.axis(axis)
    plt.colorbar()
    plt.show()        
        



def urlencoded_jpeg_img(a):
    ''' 
    returns the string of an html img tag with the urlencoded jpeg of 'a'
    supports monochrome (shape = (N,M,1) or (N,M))
    and color arrays (N,M,3)  
    '''
    fmt='jpeg'

    import PIL
    
    from io import BytesIO
    import IPython.display
    import numpy as np
    f = BytesIO()
    import base64
    PIL.Image.fromarray(np.uint8(a).squeeze() ).save(f, fmt)
    x =  base64.b64encode(f.getvalue())
    return '''<img src="data:image/jpeg;base64,{}&#10;"/>'''.format(x.decode())
    # display using IPython.display.HTML(retval)
    
       
### initialize gallery
        
gallery_style_base = """
    <style>
.gallery2 {
    position: relative;
    height: 650px; }
.gallery2 .index {
    padding: 0;
    margin: 0;
    list-style: none; }
.gallery2 .index li {
    margin: 0;
    padding: 0;
    float: left;}
.gallery2 .index a { /* gallery2 item title */
    display: block;
    background-color: #EEEEEE;
    border: 1px solid #FFFFFF;
    text-decoration: none;
    width: 1.9em;
    padding: 6px; }
.gallery2 .index a span { /* gallery2 item content */
    display: block;
    position: absolute;
    left: -9999px; /* hidden */
    top: 0em;
    padding-left: 0em; }
.gallery2 .index a span img{ /* gallery2 item content */
    max-width: 100%;
    }
.gallery2 .index li:first-child a span {
    top: 0em;
    left: 10.5em;
    z-index: 99; }
.gallery2 .index a:hover {
    border: 1px solid #888888; }
.gallery2 .index a:hover span {
    left: 10.5em;
    z-index: 100; }
</style>
    """

  
def display_gallery(image_urls, image_labels=None):
    '''
    image_urls can be a list of urls 
    or a list of numpy arrays
    image_labels is a list of strings
    '''
    from  IPython.display import HTML  
    import numpy as np

    
    gallery_template = """
    <div class="gallery2">
        <ul class="index">
            {}
        </ul>
    </div>
    """
    
    li_template = """<li><a href="#">{}<span style="background-color: white;  " ><img src="{}" />{}</span></a></li>"""
    li_template_encoded = """<li><a href="#">{}<span style="background-color: white;  " >{}{}</span></a></li>"""

    li = ""
    idx = 0
    for u in image_urls:
        if image_labels:
            label = image_labels[idx]
        else:
            label = str(idx)
        if type(u) == str:
            li = li + li_template.format( idx, u, label)
        elif type(u) == np.ndarray:
            li = li + li_template_encoded.format( idx, urlencoded_jpeg_img(u.clip(0,255) ), label)

        idx = idx + 1
        
    source = gallery_template.format(li)
    
    display(HTML( source ))
    display(HTML( gallery_style_base ))

    return 
    

    
def overprintText(im,imout,text,textRGBA=(255,255,255,255)):
    '''
    prints text in the upper left corner of im (filename) 
    and writes imout (filename)
    '''
    try:
        import Image
    except ImportError:
        from PIL import Image
    
    from PIL import ImageDraw, ImageFont
    # get an image
    base = Image.open(im).convert('RGBA')

    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))

    # get a font
    #    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(txt)

    # draw text
    d.text((1,1), text,  fill=tuple(textRGBA))
    out = Image.alpha_composite(base, txt)

    out.save(imout)

    
    
def display_patches(mb):
    '''
    disaplay all images in a list as patches in a squared figure
    Args:
        mb is a list or a numpy array of images
    '''
    # display all the patches in an array of images
    import matplotlib.pyplot as plt  
    import numpy as np
    
    plt.figure(figsize=(12, 12))
    i=0

    L = len(mb)
    M = np.floor(np.sqrt(L))
    N = np.ceil (L/M)

    for j in range(L):
        plt.subplot(M, N, i + 1)
        plt.imshow(np.array(mb[j]).squeeze(), cmap=plt.cm.gray,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
        i+=1
    plt.show()

