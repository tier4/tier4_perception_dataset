# Tools Overview

This document offers a overview of the scripts employed in the conversion process for the localization evaluation.  
Prior to reading this document, we recommend referring to the [tools overview](./tools_overview.md) for a comprehensive description of the entire repository.

## raw rosbag2 to T4 format for localization evaluation

input: rosbag2
output: T4 format data (for localization evaluation)

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_rosbag2_to_localization_evaluation.yaml
```
