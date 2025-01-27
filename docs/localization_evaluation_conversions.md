# Tools Overview

This document provides an overview of the scripts used in the conversion process for localization evaluation.  
Before reading this document, we recommend referring to the [Tools Overview](./tools_overview.md) for a comprehensive description of the entire repository.

## raw rosbag2 to T4 format for localization evaluation


This function converts raw Rosbag2 data (captured from vehicles) into the T4 format.  
The resulting T4 dataset contains only the following components:  
- Annotation JSON files under the `annotation` directory  
- Dummy PCD data under the `data` directory  
- Topic-filtered Rosbag files under the `input_bag` directory  

### Input  
- Raw Rosbag2 file  

### Output  
- T4 format dataset (for localization evaluation) with filtered input bags  

```bash
python -m perception_dataset.convert --config config/rosbag2_to_t4/convert_rosbag2_to_localization_evaluation.yaml
```
