# Tools Overview

This document provides an overview of the scripts used in the conversion process for localization evaluation.  
Before reading this document, we recommend referring to the [Tools Overview](./tools_overview.md) for a comprehensive description of the entire repository.

## rosbag2 to T4 format for localization evaluation

This function converts Rosbag2 data into the T4 format.  
The resulting T4 dataset contains only the following components:

- **Annotation JSON files**: Stored under the `annotation` directory
- **Dummy PCD data**: Stored under the `data` directory
- **Rosbag files**: Stored under the `input_bag` directory (identical to the specified Rosbag files)

### Input

- Rosbag2 file

### Output

- T4 format dataset (for localization evaluation)

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_rosbag2_to_localization_evaluation.yaml
```
