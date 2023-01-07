# Line Detection Algorithm

Line detection algorithm using two methods RANSAC and Hough Transform


Custom parameters:
```
python3 LineDetection.py np (float: hessian threshold) 
                            (float: distance threshold)
                            (float: required inliers)
                            (int: dim. of bin accum.)
                            (str: image path)
```

Default parameters (gauss std=2, gauss threshold=30):

```
python3 LineDetection.py p 0 0 0 0 (str: image path)
```

### Output:

Results including both RANSAC and Hough outputs are saved as png called ”ld-[epoch].png”
