import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from io import BytesIO
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stacking_algorithm_sample import StackingMethod, StackingAlgorithm


def save_to_json(data, filename):
    """Save data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def load_from_json(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def generate_boxes(num_boxes, x_range, y_range, z_range, sizes=None):
    if sizes is None:
        # sizes가 제공되지 않으면 x_range, y_range, z_range를 사용하여 무작위 크기를 생성
        sizes = [
            (
                np.random.randint(x_range[0], x_range[1]),
                np.random.randint(y_range[0], y_range[1]),
                np.random.randint(z_range[0], z_range[1])
            ) for _ in range(num_boxes)
        ]
    else:
        # sizes가 제공되면 그 크기를 사용
        if len(sizes) != num_boxes:
            raise ValueError("The length of sizes must match num_boxes")
    
    # 상자의 ID와 크기를 사전 형식으로 생성
    boxes = [{'box_id': index, 'box_size': size} for index, size in enumerate(sizes)]
    return boxes

def is_point_in_box(point, box_corners, z_range):
    """Check if a point is inside a given 3D box defined by its corners and z range."""
    x, y, z = point
    x_range = [np.min(box_corners[:, 0]), np.max(box_corners[:, 0])]
    y_range = [np.min(box_corners[:, 1]), np.max(box_corners[:, 1])]
    return x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]

def check_overlap_rotation(rotated_corners1, z_range1, rotated_corners2, z_range2):
    """Check if two rotated 3D boxes overlap."""
    # Check if any corner of the first box is inside the second box
    for corner in rotated_corners1:
        if is_point_in_box((corner[0], corner[1], z_range1[0]), rotated_corners2, z_range2) or \
           is_point_in_box((corner[0], corner[1], z_range1[1]), rotated_corners2, z_range2):
            return True
    # Check if any corner of the second box is inside the first box
    for corner in rotated_corners2:
        if is_point_in_box((corner[0], corner[1], z_range2[0]), rotated_corners1, z_range1) or \
           is_point_in_box((corner[0], corner[1], z_range2[1]), rotated_corners1, z_range1):
            return True
    return False

def rotate_box_corners(x_center, y_center, width, length, angle):
    """Rotate the box corners according to the given angle."""
    radians = np.deg2rad(angle)
    cos_angle = np.cos(radians)
    sin_angle = np.sin(radians)

    # Define the original corners based on the center
    corners = np.array([
        [-width / 2, -length / 2],
        [width / 2, -length / 2],
        [width / 2, length / 2],
        [-width / 2, length / 2]
    ])
    
    corners = np.ceil(corners)

    # Rotate corners
    rotated_corners = np.dot(corners, np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]))

    # Translate corners back to the center position
    rotated_corners[:, 0] += x_center
    rotated_corners[:, 1] += y_center

    return rotated_corners

def stacking_check_and_visualize(placements, boxes, cubic_range, result_folder_name, result_file_name="stacking_animation"):
    """Visualize the result of stacking boxes in 3D considering z-axis rotation."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, cubic_range[0]])
    ax.set_ylim([0, cubic_range[1]])
    ax.set_zlim([0, cubic_range[2] + 30])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Box Stacking')
    
    total_volume = 0
    stacking_number = 0
    placements_check = []
    used_buffers = []
    frames = []
    ctr = 1
    is_valid = True

    for place_box in placements:
        
        box_id = place_box["box_id"]

        if place_box['pallet_id'] == 1:  # 적재 팔레트
            if box_id in used_buffers:
                used_buffers.remove(box_id)

            width, length, height = boxes[box_id]["box_size"]
            angle = place_box["box_rot"]

            x_center = place_box["box_loc"][0]
            y_center = place_box["box_loc"][1]
            z_center = place_box["box_loc"][2]

            rotated_corners = rotate_box_corners(x_center, y_center, width, length, angle)

            # Get the bounding box of the rotated corners
            x_min, y_min = np.min(rotated_corners, axis=0)
            x_max, y_max = np.max(rotated_corners, axis=0)

            x = x_min
            y = y_min
            z = z_center
            
            # Recalculate width and length after rotation
            width = x_max - x_min
            length = y_max - y_min            
            
            faces = [
                [list(rotated_corners[0]) + [z], list(rotated_corners[1]) + [z], list(rotated_corners[1]) + [z + height], list(rotated_corners[0]) + [z + height]],
                [list(rotated_corners[1]) + [z], list(rotated_corners[2]) + [z], list(rotated_corners[2]) + [z + height], list(rotated_corners[1]) + [z + height]],
                [list(rotated_corners[2]) + [z], list(rotated_corners[3]) + [z], list(rotated_corners[3]) + [z + height], list(rotated_corners[2]) + [z + height]],
                [list(rotated_corners[3]) + [z], list(rotated_corners[0]) + [z], list(rotated_corners[0]) + [z + height], list(rotated_corners[3]) + [z + height]],
                [list(rotated_corners[0]) + [z], list(rotated_corners[1]) + [z], list(rotated_corners[2]) + [z], list(rotated_corners[3]) + [z]],
                [list(rotated_corners[0]) + [z + height], list(rotated_corners[1]) + [z + height], list(rotated_corners[2]) + [z + height], list(rotated_corners[3]) + [z + height]]
            ]
            face_color = [random.random() for _ in range(3)]
            ax.add_collection3d(Poly3DCollection(faces, facecolors=face_color, linewidths=1, edgecolors='r', alpha=.25))
            
            xaxis_violate = np.any(rotated_corners[:, 0] < 0) or np.any(rotated_corners[:, 0] > cubic_range[0])
            yaxis_violate = np.any(rotated_corners[:, 1] < 0) or np.any(rotated_corners[:, 1] > cubic_range[1])
            zaxis_violate = z + height > cubic_range[2]
            
            if xaxis_violate or yaxis_violate or zaxis_violate:
                print(f"Box with ID {box_id} is out of bound. Stopping stacking.")
                is_valid = False
            
            if any(check_overlap_rotation(rotated_corners, [z, z + height], p_corners, [p_z, p_z + p_height]) for (p_corners, p_z, p_height) in placements_check):
                print(f"Box with ID {box_id} overlaps with another box. Stopping stacking.")
                is_valid = False

            if is_valid:
                placements_check.append((rotated_corners, z, height))
                total_volume += width * length * height
                stacking_number += 1
        
        if place_box['pallet_id'] == 2:  # 버퍼 팔레트
            used_buffers.append(box_id)
            
            buffer_area = sum(
                boxes[used_box_id]["box_size"][0] * boxes[used_box_id]["box_size"][1]
                for used_box_id in used_buffers
            )
                
            print("The rest of buffer area is :" + str(cubic_range[0] * cubic_range[1] - buffer_area))
            if buffer_area > cubic_range[0] * cubic_range[1]:
                is_valid = False                
            
            if len(used_buffers) > 5:
                print(f"Buffer with ID {box_id} overlaps with another box. Stopping stacking.")
                is_valid = False
            
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

        if not is_valid:
            break

        ctr += 1
    
    stacking_rate = total_volume / (cubic_range[0] * cubic_range[1] * cubic_range[2]) * 100
    
    if frames:
        gif_path = os.path.join(result_folder_name, f'{result_file_name}.gif')    
        imageio.mimsave(gif_path, frames, format='GIF', duration=0.5)
    
    plt.show()
    
    return stacking_rate, stacking_number


def stacking_check_and_visualize1(placements, boxes, cubic_range, result_folder_name, result_file_name="stacking_animation.html"):
    frames = []
    placements_check = []
    total_volume = 0
    stacking_number = 0
    used_buffers = []
    fig = go.Figure()

    for place_box in placements:
        box_id = place_box["box_id"]
        if place_box['pallet_id'] == 1:
            if box_id in used_buffers:
                used_buffers.remove(box_id)

            width, length, height = boxes[box_id]["box_size"]
            angle = place_box["box_rot"]

            x_center = place_box["box_loc"][0]
            y_center = place_box["box_loc"][1]
            z_center = place_box["box_loc"][2]

            rotated_corners = rotate_box_corners(x_center, y_center, width, length, angle)

            x_min, y_min = np.min(rotated_corners, axis=0)
            x_max, y_max = np.max(rotated_corners, axis=0)

            x = x_min
            y = y_min
            z = z_center

            width = x_max - x_min
            length = y_max - y_min

            vertices = [
                [x_min, y_min, z],
                [x_max, y_min, z],
                [x_max, y_max, z],
                [x_min, y_max, z],
                [x_min, y_min, z + height],
                [x_max, y_min, z + height],
                [x_max, y_max, z + height],
                [x_min, y_max, z + height]
            ]

            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[3], vertices[0], vertices[4], vertices[7]],
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]]
            ]

            if np.any(rotated_corners[:, 0] < 0) or np.any(rotated_corners[:, 0] > cubic_range[0]) or \
               np.any(rotated_corners[:, 1] < 0) or np.any(rotated_corners[:, 1] > cubic_range[1]) or \
               z + height > cubic_range[2]:
                break

            if any(check_overlap_rotation(rotated_corners, [z, z + height], p_corners, [p_z, p_z + p_height]) for (p_corners, p_z, p_height) in placements_check):
                break

            placements_check.append((rotated_corners, z, height))
            total_volume += width * length * height
            stacking_number += 1

            fig.add_trace(
                go.Mesh3d(
                    x=[v[0] for v in vertices],
                    y=[v[1] for v in vertices],
                    z=[v[2] for v in vertices],
                    i=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                    j=[1, 2, 3, 2, 3, 0, 3, 0, 1, 0, 1, 2],
                    k=[4, 5, 6, 5, 6, 7, 6, 7, 4, 7, 4, 5],
                    opacity=0.50,
                    color='blue'
                )
            )

        if place_box['pallet_id'] == 2:
            used_buffers.append(box_id)

            buffer_area = sum(
                boxes[used_box_id]["box_size"][0] * boxes[used_box_id]["box_size"][1]
                for used_box_id in used_buffers
            )

            if buffer_area > cubic_range[0] * cubic_range[1]:
                break

            if len(used_buffers) > 5:
                break

        frames.append(go.Frame(data=fig.data))

    fig.update(frames=frames)
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                 "label": "Play",
                 "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                 "label": "Pause",
                 "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )
    gif_path = os.path.join(result_folder_name, f'{result_file_name}.html')
    fig.write_html(gif_path)
    return total_volume / (cubic_range[0] * cubic_range[1] * cubic_range[2]) * 100, stacking_number

if __name__ == "__main__":
    data_dir = "sample_json/"
    os.makedirs(data_dir, exist_ok=True)

    ###############################################################################
    ## 랜덤 박스 생성
    num_small_boxes, num_large_boxes = 75, 5
    boxes_small = generate_boxes(num_small_boxes, x_range=[200,500], y_range=[200,500], z_range=[200,500])    
    boxes_large = generate_boxes(num_large_boxes, x_range=[500,700], y_range=[500,700], z_range=[500,700])
    boxes = boxes_small + boxes_large
    save_to_json(boxes, data_dir + 'random_boxes.json')    
        
    ###############################################################################
    ## 테스트 코드 
    loaded_boxes = load_from_json(data_dir + 'random_boxes.json')    
    cubic_range = [1100, 1100, 1800]

    stack_algo = StackingAlgorithm(loaded_boxes, cubic_range)
    methods = \
        [
            # StackingMethod.PALLET_ORIGIN_OUT_OF_BOUND,
            # StackingMethod.PALLET_Z_EXCEED           ,
            StackingMethod.PALLET_X_EXCEED           ,
            # StackingMethod.PALLET_Y_EXCEED           ,
            # StackingMethod.PALLET_CENTER             ,
            # StackingMethod.PALLET_CENTER_ROT         ,
            # StackingMethod.PALLET_CORNER_XY_AXIS     ,
            # StackingMethod.PALLET_CORNER_Z_AXIS
            # StackingMethod.PALLET_STACK_ALL
        ]
    
    ## Json 파일로 저장
    for method in methods:
       print(method)
       save_to_json(stack_algo.stack(method), data_dir + f'{method.name}.json')
    
    ###############################################################################
    ## 검증 및 시각화
    for method in methods:
        print(f'Validate and visualize "{method.name}" stacking algorithm')
        loaded_placements = load_from_json(data_dir + f'{method.name}.json')

        stacking_rate, stacking_number = \
            stacking_check_and_visualize(
                loaded_placements,
                loaded_boxes,
                cubic_range,
                data_dir,
                method.name)
    
        print('Stacking rate: {:.2f}%'.format(stacking_rate)) 
        print('Stacking box number:', stacking_number)
        print()
    ###############################################################################
    print('done')