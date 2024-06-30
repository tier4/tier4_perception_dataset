## Sample usage

```py

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

### Rendering scene

```py
>>> scene_token = t4.scene[0].token
>>> t4.render_scene(scene_token)
```

<!-- TODO: add rendering result video -->
