VideoCap
========

Setup
-----

### Basic Dependencies

```
pip install -r requirements.txt
```

### Python PATH

```
git submodule update --init --recursive
add2virtualenv .
add2virtualenv coco-caption
```

```
cd coco/PythonAPI
python setup.py install
```

Prepare Data
------------

* `dataset/mscoco/`
* `dataset/lsmdc/`

To make symbolic link:

```
ln -sf /data/common_datasets/captiongaze/ dataset/lsmdc/
```

See README.md for details.


Running Tests (Optional)
------------------------

```
green
green videocap/tests/test_datasets.py   # or green test_datasets.py in the directory
```
