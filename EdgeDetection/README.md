# Canny Edge Detection Algorithm

Edge detection algorithm with multiple steps done in pure numpy, see [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector) for more on implementation.


Custom parameters:
```
python3 EdgeDetection.py np (float: gaussian std) 
                            (float: gradient threshold) 
                            (str: image path/s)+
```

Default parameters (gauss std=2, gauss threshold=30):

```
python3 EdgeDetection.py p 0 0 (str: image path/s)+
```

### Output:

Results are saved as pngs called ```[filename]-[epoch].png``` in new file called ```ed-[epoch]```

