from setuptools import find_packages, setup

package_name = 'belief_updater'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','PyYAML'],
    zip_safe=True,
    maintainer='edo',
    maintainer_email='edoardo.barutta00@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'belief_updater_service = belief_updater.belief_updater_service:main',
        ],
    },
)
