from setuptools import find_packages, setup

package_name = 'behaviour_tree'

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
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
             'behaviour_tree_noROS_MF = behaviour_tree.behaviour_tree_noROS_MF:main',
             'behaviour_tree = behaviour_tree.behaviour_tree:main',
        ],
    },
)
