docker build -t suprref .
docker run -it -v %cd%:/home/promoter/suprref suprref
echo Build Done
