import pybullet as p
p.connect(p.DIRECT)
robot_id = p.loadURDF("/home/edo/thesis/ur5e_urdf/urdf/ur5e_complete.urdf")
num_joints = p.getNumJoints(robot_id)

# Create mapping from joint index to child link name
child_link_names = {}
for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    child_link_names[i] = info[12].decode('utf-8')  # This joint's parent becomes the next joint's child

print("\n" + "="*50)
print("Joint name: Parent link <-> Child link")
print("-" * 50)

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    joint_name = info[1].decode('utf-8')  # joint name
    parent_link_name = info[12].decode('utf-8')  # parent link name
    child_link_index = info[16]  # child link index
    
    # Get child link name
    if child_link_index >= 0 and child_link_index < num_joints:
        child_link_name = child_link_names.get(child_link_index, f"link_{child_link_index}")
    else:
        child_link_name = "N/A"
    
    print(f"{joint_name}: {child_link_name} <-> {parent_link_name}")

print("\n" + "="*50)
print("ALL LINKS IN THE ROBOT:")
print("="*50)

# Get base link name
base_link_name = p.getBodyInfo(robot_id)[1].decode('utf-8')
print(f"Base link: {base_link_name}")

print("\n" + "="*50)
print("COMPLETE LINK LIST:")
print("="*50)

# Create a set of all unique links
all_links = set()
all_links.add(base_link_name)

for i in range(num_joints):
    info = p.getJointInfo(robot_id, i)
    parent_link = info[12].decode('utf-8')
    all_links.add(parent_link)

# Print all links in alphabetical order
for link in sorted(all_links):
    print(f"- {link}")