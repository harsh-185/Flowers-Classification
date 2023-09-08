import tarfile
import os
tar = tarfile.open('17flowers.tgz', 'r')
tar.extractall()
tar.close()
