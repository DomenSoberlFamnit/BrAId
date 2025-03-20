# Project BrAId
A collection of Python scripts to extract and prepare data from SiWIM database, and conduct deep learning for axle group detection.

## Description of folders
- data: extraction of data and creation of training/testing datasets.
- metadata: a collection of manually prepared data filters and individual corrections of selected data samples.
- training: scripts for training TensorFlow-based CNN models.
- testing: scripts to test the trained models.
- results: results of testing.
- statistics: scripts to extract from the results the statistical representations.
- tools: various tools for data manipulation and insights.

## Usage

Only the trained models can be used by external users that don't have access to the internal libraries and databases.

### Installation

`pip install BrAId@git+https://github.com/DomenSoberlFamnit/BrAId`

### Minimal example

```
from PIL import Image
import braid

img = Image.open('truck_sample.jpg')
results = braid.axle_groups_from_image(image=img, site='sentvid')
print(results)
```
