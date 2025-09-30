from setuptools import find_packages, setup

package_name = 'vision_service'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='edo',
    maintainer_email='edoardo.barutta00@gmail.com',
    description='Perception service using OAK-D and CNN to return centroid and optional volume',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vis_service = vision_service.vis_service:main'
        ],
    },
)
