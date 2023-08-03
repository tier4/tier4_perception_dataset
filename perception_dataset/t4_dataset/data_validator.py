
from loguru import logger
from nuscenes.nuscenes import NuScenes
from perception_dataset.t4_dataset.format_validator import _logger_wrapper

@_logger_wrapper
def validate_data_hz(nusc: NuScenes):
    def get_first_sample_data_list():
        sample_data_list = []
        for sample_data in nusc.sample_data:
            if sample_data["prev"] == "":
                sample_data_list.append(sample_data)
        return sample_data_list

    for sample_data in get_first_sample_data_list():
        if "is_valid" in sample_data and not sample_data["is_valid"]:
            continue
        first_filename: str = sample_data["filename"]
        first_timestamp: int = sample_data["timestamp"]
        sample_data_counter: int = 0
        while sample_data["next"]:
            sample_data_counter += 1
            sample_data = nusc.get("sample_data", sample_data["next"])

        data_duration_sec: float = float(sample_data["timestamp"] - first_timestamp) * 1e-6
        data_hz: float = sample_data_counter / data_duration_sec

        logger.info(f"{first_filename}")
        logger.info(f"Duration: {data_duration_sec} sec")
        logger.info(f"Hz: {data_hz}")

        assert data_hz > 9.0
