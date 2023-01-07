# Two View Image Alignment

Two view image alignment algorithm, stiches two images together via Harris and NCC correspondence detection and Affine transformation approximation via RANSAC


Custom parameters:
```
python3 TwoViewAlignment.py np (int: max_features)
                               (int: similiarity_window)
                               (int: correspondence)
                               (str: image A path)
                               (str: image B path)

```

Default parameters (max_features=1000, similarity_window=12, correspondences=20):
```
python3 TwoViewAlignment.py p 0 0 0 (str: image left)
                                 (str: image right)
```

### Output:

Results including Harris, NCC correspondences, and Alignment outputs are saved as
pngs in directory ```tva-[epoch]```

### Note:

Images MUST be RGB format (N x M x 3) and both the same size
