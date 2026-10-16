from glob import glob

from setuptools import find_packages, setup


package_name = 'tb3_localization_lab'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml', 'LICENSE']),

        # Install the experiment's launch files and configuration assets.
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        (
            'share/' + package_name + '/models/burger_ground_truth',
            glob('models/burger_ground_truth/*.sdf'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='txc632@student.bham.ac.uk',
    description='Wheel odometry and ground-truth evaluation for TurtleBot3.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'wheel_odometry = tb3_localization_lab.wheel_odometry:main',
            'encoder_model = tb3_localization_lab.encoder_model:main',
        ],
    },
)