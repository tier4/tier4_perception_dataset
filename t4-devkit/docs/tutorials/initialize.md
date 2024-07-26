## Initialize `Tier4` API

---

`Tier4` API expects the following dataset directly structure:

```shell
data/tier4/
├── annotation ...contains `*.json` files.
├── data
│   ├── CAM_BACK
│   ├── CAM_BACK_LEFT
│   ├── CAM_BACK_RIGHT
│   ├── CAM_FRONT
│   ├── CAM_FRONT_LEFT
│   ├── CAM_FRONT_RIGHT
│   ├── LIDAR_CONCAT
│   └── ...Other sensor channels
...
```

You can initialize a `Tier4` instance as follows:

```python

>>> from t4_devkit import Tier4

>>> t4 = Tier4("annotation", "data/tier4/", verbose=True)
======
Loading T4 tables in `annotation`...
Reverse indexing...
Done reverse indexing in 0.007 seconds.
======
13 attribute
7 calibrated_sensor
8 category
2524 ego_pose
106 instance
1 log
1 map
72 sample
2390 sample_annotation
2524 sample_data
4 visibility
7 sensor
1 scene
1326 object_ann
0 surface_ann
0 keypoint
Done loading in 0.061 seconds.
======
```

## Accessing to Fields

---

### `scene`

```python
>>> my_scene = t4.scene[0]
```

### `sample`

`sample` is an annotated keyframe of a scene at a given timestamp.

```python
>>> first_sample_token = my_scene.first_sample_token
>>> my_sample = t4.get("sample", first_sample_token)
```

You can access to the `sample_data` associated with this `sample`.
`sample.data` returns a `dict` object consists of `{str: <SAMPLE_DATA_TOKEN (str)>}`.

```python
>>> my_sample.data
```

### `sample_data`

`sample_data` is references to a family of data that is collected from specific sensors.

```python
>>> sensor = "CAM_FRONT"
>>> t4.get("sample_data", my_sample.data[sensor])
```

### `sample_annotation`

`sample_annotation` refers to any 3D bounding box in a corresponding `sample`.
All location data is given with respect to the global coordinate system.
You can access to the list of `sample_annotation` tokens with `sample.ann_3ds: list[str]`.

```python
>>> my_annotation_token = my_sample.ann_3ds[0]
>>> t4.get("sample_annotation", my_annotation_token)
```

### `instance`

Each annotated object is instanced to be tracked.

```python
>>> t4.instance
```

### `category`

A `category` is the object assignment of an annotation.

```python
>>> t4.category
```

### `attribute`

An `attribute` is a property of an instance that may change throughout different parts of a scene while `category` remains the same.

```python
>>> t4.attribute
```

### `visibility`

`visibility` is defined as the fraction of pixels of a particular annotation that are visible over the 6 camera feeds.

```python
>>> t4.visibility
```

<!-- prettier-ignore-start -->
??? WARNING
    Expected `level` values in `visibility` are as below:

    <!-- markdownlint-disable MD046 -->
    ```yaml
    - full    : No occlusion for the object.
    - most    : Object is occluded, but by less than 50%.
    - partial : Object is occluded, but by more than 50%.
    - none    : Object is 90-100% occluded and no points/pixels are visible.
    ```

    Following old formats are also supported but deprecated:

    <!-- markdownlint-disable MD046 -->
    ```yaml
    - v80-100 : full
    - v60-80  : most
    - v40-60  : partial
    - v0-40   : none
    ```

    If input level does not satisfy any above cases, `VisibilityLevel.UNAVAILABLE` will be assigned.
<!-- prettier-ignore-end -->

### `sensor`

T4 dataset consists of several type of sensors.
The supported sensor modalities and channels are defined in `t4_devkit/schema/tables/sensor.py`.

```python
>>> t4.sensor
```

### `calibrated_sensor`

`calibrated_sensor` consists of the definition of a calibration of a particular sensor based on a vehicle.

```python
>>> t4.calibrated_sensor
```

Note that the `translation` and `rotation` parameters are given with respect to the ego vehicle body frame.

### `ego_pose`

`ego_pose` contains information about the `translation` and `rotation` of the ego vehicle, with respect to the global coordinate system.

```python
>>> t4.ego_pose
```
