from setuptools import find_packages, setup

package_name = "t4dataset_rosbag_converter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jacob Lambert",
    maintainer_email="jacob@example.com",
    description="Offline rosbag converter using Nebula and Autoware pointcloud libraries.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "convert_raw_rosbag = t4dataset_rosbag_converter.cli:main",
            "smoke_imports = t4dataset_rosbag_converter.smoke_imports:main",
        ],
    },
)
