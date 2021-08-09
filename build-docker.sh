#!/bin/bash
docker build -t suprref .
docker run -it -v `pwd`:/home/promoter/suprref suprref
echo Build Done