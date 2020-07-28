#!/bin/bash

mkdir ckpt/ && cd ckpt/
wget https://www.dropbox.com/s/9rf37el19wbrydb/jazz-trsfmr-B-loss0.25.data-00000-of-00001?dl=0 -O jazz-trsfmr-B-loss0.25.data-00000-of-00001
wget https://www.dropbox.com/s/t8b916f6ftt249z/jazz-trsfmr-B-loss0.25.index?dl=0 -O jazz-trsfmr-B-loss0.25.index
wget https://www.dropbox.com/s/q41lkoym4n7d0i6/jazz-trsfmr-B-loss0.25.meta?dl=0 -O jazz-trsfmr-B-loss0.25.meta
cd ../