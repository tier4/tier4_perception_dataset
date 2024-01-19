# Data Interpolation

Interpolate 3D annotations based on LiDAR timestamps recorded in sample data.

## Assumptions

- LiDAR and 3D annotation data is included.
- Sample and sample annotation records will be interpolated with the non-key frame timestamp corresponding to the LiDAR timestamp.

## Metadata updates

By interpolation, the following metadata will be updated.

- **scene.json**
  - Only update `description` and `nbr_samples`
    - `token` ..._No update_
    - `name` ..._No update_
    - `description` ...Insert `interpolate`
    - `log_token` ..._No update_
    - `nbr_samples` ...Update with the number of interpolated `sample`
    - `first_sample_token` ..._No update_
    - `last_sample_token` ..._No update_
- **instance.json**
  - Only update `nbr_annotations`
    - `token` ..._No update_
    - `category_token` ..._No update_
    - `instance_name` ..._No update_
    - `nbr_annotations` ...Update with the number of interpolated `sample_annotation`
    - `first_annotation_token` ..._No update_
    - `last_annotation_token` ..._No update_
- **sample_data.json**
  - Set the updated `sample_token` with the interpolated `sample` and set `is_key_frame=True`
    - `token` ..._No update_
    - `sample_token` ...If `is_key_frame=False` and LiDAR data is, add a new sample and set its token
    - `ego_pose_token` ..._No update_
    - `calibrated_sensor_token` ..._No update_
    - `filename` ..._No update_
    - `fileformat` ..._No update_
    - `width` ..._No update_
    - `height` ..._No update_
    - `timestamp` ..._No update_
    - `is_key_frame` ...Set `True` if there is a corresponding interpolated `sample`
    - `next` ..._No update_
    - `prev` ..._No update_
- **sample.json**
  - Add a new record with the interpolated `timestamp`.
  - Update `next/prev` token in the original record if there are any new records around at its `timestamp`.
    - `token` ..._No update_
    - `timestamp` ..._No update_
    - `scene_token` ..._No update_
    - `next` ...Update if there is a new next record
    - `prev` ...Update if there is a new previous record
- **sample_annotation.json**
  - Add a new record interpolating `translation/rotation/velocity/acceleration` with the corresponding timestamp
  - Update `next/prev` token in the original record if there are any new records around at its `timestamp`.
    - `token` ..._No update_
    - `sample_token` ...Fill with the corresponding token of `sample`
    - `instance_token` ..._No update_
    - `attribute_tokens` ..._No update_
    - `visibility_token` ...Set the same value with the latest previous original `sample_annotation` for the same instance
    - `translation` ...Interpolate with `CUBIC_SPLINE`
    - `velocity` ...Interpolate with `CUBIC_SPLINE` if not `None`
    - `acceleration` ...Interpolate with `CUBIC_SPLINE` if not `None`
    - `size` ...Set the same value with the latest previous original `sample_annotation` for the same instance
    - `rotation` ...Interpolate with `SLERP`
    - `num_lidar_pts` ...Set the same value with the latest previous original `sample_annotation` for the same instance
    - `num_radar_pts` ...Set the same value with the latest previous original `sample_annotation` for the same instance
    - `next` ...Update if there is a new next record
    - `prev` ...Update if there is a new previous record

## Quick Start

```bash
python3 -m perception_dataset.convert --config <INTERPOLATION_CONFIG>
```

### Config Settings

```yaml
task: interpolate                   ...Task name
description:
  scene: "Interpolate"              ...Scene description
conversion:
  input_base: ./data/t4_dataset     ...Input base directory path
  output_base: ./data/interpolate   ...Output base directory path
```

## Check Interpolation Result

```bash
python3 perception_dataset/t4_dataset/data_interpolator.py <INTERPOLATED_DATA_BASE>
```
