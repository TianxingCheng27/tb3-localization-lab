"""Add independent simulator ground truth to the copied configuration."""

from pathlib import Path
import xml.etree.ElementTree as ET


PLUGIN_NAME = 'gz::sim::systems::PosePublisher'

PLUGIN_XML = '''
    <!-- Independent simulator pose, used only for evaluation. -->
    <plugin filename="gz-sim-pose-publisher-system"
            name="gz::sim::systems::PosePublisher">
      <publish_model_pose>true</publish_model_pose>
      <publish_nested_model_pose>false</publish_nested_model_pose>
      <publish_link_pose>false</publish_link_pose>
      <publish_visual_pose>false</publish_visual_pose>
      <publish_collision_pose>false</publish_collision_pose>
      <publish_sensor_pose>false</publish_sensor_pose>
      <use_pose_vector_msg>false</use_pose_vector_msg>
      <static_publisher>false</static_publisher>
      <update_frequency>50</update_frequency>
      <topic>/ground_truth/pose</topic>
    </plugin>

'''

BRIDGE_ENTRY = '''
# Independent world pose, used only for evaluation.
- ros_topic_name: /ground_truth/pose
  gz_topic_name: /ground_truth/pose
  ros_type_name: geometry_msgs/msg/PoseStamped
  gz_type_name: gz.msgs.Pose
  direction: GZ_TO_ROS
'''


def main():
    # Locate this source package from the script's location.
    package_dir = Path(__file__).resolve().parent.parent
    model_path = package_dir / 'models/burger_ground_truth/model.sdf'
    world_path = package_dir / 'worlds/odometry_test.world'
    bridge_path = package_dir / 'config/turtlebot3_bridge.yaml'

    # Check the model structure before changing its configuration.
    model_text = model_path.read_text(encoding='utf-8')
    root = ET.fromstring(model_text)
    models = root.findall('model')

    if len(models) != 1 or models[0].find('model') is not None:
        raise ValueError('Expected one model without nested models.')

    model = models[0]
    footprint = model.find("link[@name='base_footprint']")

    if footprint is None:
        raise ValueError('The model must contain base_footprint.')

    if footprint.find('pose') is not None:
        raise ValueError('Inspect the base_footprint transform first.')

    world = ET.parse(world_path).getroot().find('world')
    if world is None or world.get('name') != 'default':
        raise ValueError('Expected the world to be named default.')

    # Add the plugin once; check its settings on subsequent runs.
    plugin = model.find(f"plugin[@name='{PLUGIN_NAME}']")

    if plugin is None:
        closing_tag = '  </model>'
        if model_text.count(closing_tag) != 1:
            raise ValueError('Could not identify the model closing tag.')

        model_text = model_text.replace(
            closing_tag, PLUGIN_XML + closing_tag,
        )
    else:
        expected = ET.fromstring(PLUGIN_XML)
        actual_values = {
            child.tag: (child.text or '').strip() for child in plugin
        }
        expected_values = {
            child.tag: (child.text or '').strip() for child in expected
        }

        if (plugin.attrib != expected.attrib
                or actual_values != expected_values):
            raise ValueError('Existing PosePublisher settings differ.')

    # Preserve existing bridges and append the ground-truth bridge once.
    ET.fromstring(model_text)
    bridge_text = bridge_path.read_text(encoding='utf-8')

    if '/ground_truth/pose' not in bridge_text:
        bridge_text = bridge_text.rstrip() + '\n' + BRIDGE_ENTRY
    elif BRIDGE_ENTRY.strip() not in bridge_text:
        raise ValueError('Inspect the existing ground-truth bridge.')

    # Write the prepared configuration into our copied files.
    model_path.write_text(model_text, encoding='utf-8')
    bridge_path.write_text(bridge_text, encoding='utf-8')

    print('Model XML check passed; world name: default.')
    print('Ground-truth PosePublisher configured at 50 Hz.')
    print('Bridge configured: /ground_truth/pose -> PoseStamped.')


if __name__ == '__main__':
    main()
