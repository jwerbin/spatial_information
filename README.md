# spatial_information
Contains code for calculating spaital information metrics

## Example use of the code

```
# Load image / array
import imageio.v3 as iio
import spatial_information as si

img = iio.imread('/path/to/my/image.png')
kspace_information = si.kspace_information(img)
```