import numpy as np
import math

from enum import Enum

class StackingMethod(Enum):
    PALLET_ORIGIN_OUT_OF_BOUND  = 0,
    PALLET_Z_EXCEED             = 1,
    PALLET_X_EXCEED             = 2,
    PALLET_Y_EXCEED             = 3,    
    PALLET_CENTER               = 4,
    PALLET_CENTER_ROT           = 5,
    PALLET_CORNER_XY_AXIS       = 6,
    PALLET_CORNER_Z_AXIS        = 7,
    PALLET_STACK_ALL            = 8


class StackingAlgorithm():
    def __init__(self, boxes:dict, pallet_size:list, box_gap:int = 5):
        self.boxes = boxes.copy()
        self.pallet_size = pallet_size.copy()
        self.stacking_interval = box_gap

    def stack(self, stacking_method:StackingMethod):
        print('stack')
        if stacking_method == StackingMethod.PALLET_ORIGIN_OUT_OF_BOUND:
            return self.stack_pallet_origin_out_of_bound()
        elif stacking_method == StackingMethod.PALLET_X_EXCEED:
            return self.stack_pallet_x_exceed()
        elif stacking_method == StackingMethod.PALLET_Y_EXCEED:
            return self.stack_pallet_y_exceed()
        elif stacking_method == StackingMethod.PALLET_Z_EXCEED:
            return self.stack_pallet_z_exceed()
        elif stacking_method == StackingMethod.PALLET_CENTER:
            return self.stack_pallet_center()
        elif stacking_method == StackingMethod.PALLET_CENTER_ROT:
            return self.stack_pallet_center(box_max_rotation=90)
        elif stacking_method == StackingMethod.PALLET_CORNER_XY_AXIS:
            return self.stack_pallet_corner_xy_axis()
        elif stacking_method == StackingMethod.PALLET_CORNER_Z_AXIS:
            return self.stack_pallet_corner_z_axis()
        elif stacking_method == StackingMethod.PALLET_STACK_ALL:
            return self.stack_all_boxes()
        
    ############################################################
    ### Algorithms##############################################
    def stack_pallet_origin_out_of_bound(self):
        result = []

        pallet_id = 1
        box = self.boxes[0]

        result.append({
            "box_id"    : box["box_id"],
            "box_loc"   : (0, 0, 0),
            "box_rot"   : 0,
            "pallet_id" : pallet_id
        })

        return result

    def stack_pallet_x_exceed(self):
        result = []

        p_x = 0
        p_w = self.pallet_size[0]
        pallet_id = 1

        for box in self.boxes:
            b_x, b_y = int(box["box_size"][0]/2), int(box["box_size"][1]/2)
            
            ## 적재된 박스의 높이가 팔레스 최대 높이에 닿을 경우
            if p_x > p_w:
                break

            result.append({
                "box_id"    : box["box_id"],
                "box_loc"   : (p_x+b_x, b_y, 0),
                "box_rot"   : 0,
                "pallet_id" : pallet_id
            })
            
            ##
            p_x += box["box_size"][0] + self.stacking_interval
        print("called")
        return result
    
    def stack_pallet_y_exceed(self):
        result = []

        p_y = 0
        p_l = self.pallet_size[1]
        pallet_id = 1

        for box in self.boxes:
            b_x, b_y = int(box["box_size"][0]/2), int(box["box_size"][1]/2)
            
            ## 적재된 박스의 높이가 팔레스 최대 높이에 닿을 경우
            if p_y > p_l:
                break

            result.append({
                "box_id"    : box["box_id"],
                "box_loc"   : (b_x, p_y+b_y, 0),
                "box_rot"   : 0,
                "pallet_id" : pallet_id
            })
            
            ##
            p_y += box["box_size"][1] + self.stacking_interval
            
        return result
    
    def stack_pallet_z_exceed(self):
        result = []

        p_z = 0
        p_h = self.pallet_size[2]
        pallet_id = 1

        for box in self.boxes:
            b_x, b_y = int(box["box_size"][0]/2), int(box["box_size"][1]/2)
            
            ## 적재된 박스의 높이가 팔레스 최대 높이에 닿을 경우
            if p_z > p_h:
                break

            result.append({
                "box_id"    : box["box_id"],
                "box_loc"   : (b_x, b_y, p_z),
                "box_rot"   : 0,
                "pallet_id" : pallet_id
            })
            
            ##
            p_z += box["box_size"][2] + self.stacking_interval
            
        return result
    
    def stack_pallet_center(self, box_max_rotation:int=0):
        result = []

        p_cen_x = int(self.pallet_size[0]/2)
        p_cen_y = int(self.pallet_size[1]/2)
        p_h =  self.pallet_size[2]

        p_stacked_z = 0 
        pallet_id = 1
        
        for box in self.boxes:
            b_id, b_h = box["box_id"], box["box_size"][2]

            ## 적재된 박스의 높이가 팔레스 최대 높이에 닿을 경우
            if p_stacked_z + b_h > p_h:
                if pallet_id == 1:
                    pallet_id = 2
                    p_stacked_z = 0 
            elif pallet_id == 2 and p_stacked_z > 0:
                break

            ##
            result.append({
                "box_id"    : b_id,
                "box_loc"   : (p_cen_x, p_cen_y, p_stacked_z),
                "box_rot"   : 0 if box_max_rotation == 0 else np.random.randint(0,box_max_rotation),
                "pallet_id" : pallet_id
            })
            
            ##
            p_stacked_z += b_h + self.stacking_interval

        return result

    def stack_pallet_corner_xy_axis(self):
        result = []
        corner_cnt = 4
        p_h =  self.pallet_size[2]

        pallet_id = 1
        corner_idx = 0          # Pallet의 각 코너이며 값은 0~3
        p_stacked_z = [0] * corner_cnt
        is_corner_vacant = [True] * corner_cnt

        for box in self.boxes:
            b_id, b_h = box["box_id"], box["box_size"][2]
            
            ## 코너 선택
            corner_idx %= 4

            ## 적재된 박스의 높이가 팔레스 최대 높이에 닿을 경우
            if p_stacked_z[corner_idx] + b_h > p_h:
                p_stacked_z[corner_idx] = -1
                is_corner_vacant[corner_idx] = False

            if sum(is_corner_vacant) == 0:
                break

            if p_stacked_z[corner_idx] == -1:
                continue

            ##
            stack_pos = \
                self.get_corner_statcking_xy_pos(
                    box_xy=box["box_size"][:2],
                    pallet_size=self.pallet_size[:2],
                    corner_idx=corner_idx
                )

            ##
            result.append({
                "box_id"    : b_id,
                "box_loc"   : (stack_pos[0], stack_pos[1], p_stacked_z[corner_idx]),
                "box_rot"   : 0,
                "pallet_id" : pallet_id
            })
            
            ##
            p_stacked_z[corner_idx] += b_h + self.stacking_interval
            corner_idx += 1
            
        return result

    def stack_pallet_corner_z_axis(self):
        result = []

        p_h =  self.pallet_size[2]

        p_stacked_z = 0 
        pallet_id = 1
        corner_idx = 0    # Pallet의 각 코너이며 값은 1~4
        
        for box in self.boxes:
            b_id, b_h = box["box_id"], box["box_size"][2]

            ## 적재된 박스의 높이가 팔레스 최대 높이에 닿을 경우
            if p_stacked_z + b_h > p_h:
                p_stacked_z = 0 
                corner_idx += 1

            if corner_idx == 4:
                break

            ##
            stack_pos = \
                self.get_corner_statcking_xy_pos(
                    box_xy=box["box_size"][:2],
                    pallet_size=self.pallet_size[:2],
                    corner_idx=corner_idx
                )

            ##
            result.append({
                "box_id"    : b_id,
                "box_loc"   : (stack_pos[0], stack_pos[1], p_stacked_z),
                "box_rot"   : 0,
                "pallet_id" : pallet_id
            })
            
            ##
            p_stacked_z += b_h + self.stacking_interval
            
        return result
    
    def stack_all_boxes(self):
        """Place boxes without overlapping, stacking along x, then y, then z."""
        placements, out_placements = [], []
        
        for box in self.boxes:
            width, length, height = box["box_size"]
            placed = False
            
            for z in range(0, self.pallet_size[2] - height + 1, self.stacking_interval):
                if placed:
                    break

                for y in range(0, self.pallet_size[1] - length + 1, self.stacking_interval):
                    if placed:
                        break

                    for x in range(0, self.pallet_size[0] - width + 1, self.stacking_interval):

                        if not self.is_overlap((x, y, z , width, length, height), placements):
                            placements.append((
                                x, y, z, 
                                width + self.stacking_interval,
                                length + self.stacking_interval,
                                height + self.stacking_interval
                                ))

                            b_x, b_y, b_z = x + math.ceil(width/2), y + math.ceil(length/2), z
                            out_placements.append({
                                "box_id": box["box_id"], 
                                "box_loc": (b_x, b_y, b_z),
                                "box_rot": 0, 
                                "pallet_id": 1
                            })

                            placed = True
                            break

        return out_placements
    ############################################################
    
    ############################################################
    ### Utils ##################################################
    def get_corner_statcking_xy_pos(self, box_xy:list, pallet_size:list, corner_idx:int):
        r'''

        :param corner_idx (int)     : 0 ..., 1 ..., 2 ..., 3 ...
        '''

        b_place_x = int(math.ceil(box_xy[0] / 2))
        b_place_y = int(math.ceil(box_xy[1] / 2))

        if corner_idx == 0:
            stack_pos = \
                [
                    b_place_x,
                    b_place_y
                ]
        elif corner_idx == 1:
            stack_pos = \
                [
                    pallet_size[0] - b_place_x,
                    b_place_y
                ]
        elif corner_idx == 2:
            stack_pos = \
                [
                    b_place_x,
                    pallet_size[1] - b_place_y
                ]
        elif corner_idx == 3:
            stack_pos = \
                [
                    pallet_size[0] - b_place_x,
                    pallet_size[1] - b_place_y
                ]

        return stack_pos
    
    def is_overlap(self, box, placements):    
        bx, by, bz, bwidth, blength, bheight = box

        for px, py, pz, pwidth, plength, pheight in placements:
            if not (bx + bwidth <= px or bx >= px + pwidth or
                    by + blength <= py or by >= py + plength or
                    bz + bheight <= pz or bz >= pz + pheight):
                return True
        
        return False
    ############################################################

    def step(self):
        """
        어떻게 액션을 수행할 지가 가장 중요
        1. 중심 좌표를 action으로 하면 너무 어려우려나 ?

        """

        placements, out_placements = [], []  # 내부 배치 정보와 최종 배치 결과를 저장할 리스트 초기화

        for box in self.boxes:  # 모든 상자에 대해 반복
            width, length, height = box["box_size"]  # 상자의 크기 정보 가져오기
            placed = False  # 상자가 배치되었는지 여부를 추적하는 플래그

            for z in range(0, self.pallet_size[2] - height + 1, self.stacking_interval):  # z축 방향으로 가능한 위치 탐색
                if placed:
                    break  # 상자가 배치된 경우 반복 종료

                for y in range(0, self.pallet_size[1] - length + 1, self.stacking_interval):  # y축 방향으로 가능한 위치 탐색
                    if placed:
                        break  # 상자가 배치된 경우 반복 종료

                    for x in range(0, self.pallet_size[0] - width + 1, self.stacking_interval):  # x축 방향으로 가능한 위치 탐색
                        if not self.is_overlap((x, y, z, width, length, height), placements):  # 현재 위치에서 상자가 겹치지 않는지 확인
                            # 상자가 겹치지 않으면 placements에 위치 정보 추가
                            placements.append((
                                x, y, z,  # 상자의 좌표
                                width + self.stacking_interval,  # 상자의 너비에 간격 추가
                                length + self.stacking_interval,  # 상자의 길이에 간격 추가
                                height + self.stacking_interval  # 상자의 높이에 간격 추가
                            ))

                            # 최종 배치 결과에 상자 정보 추가
                            b_x, b_y, b_z = x + math.ceil(width / 2), y + math.ceil(length / 2), z  # 상자의 중앙 좌표 계산
                            out_placements.append({
                                "box_id": box["box_id"],  # 상자의 ID
                                "box_loc": (b_x, b_y, b_z),  # 상자의 위치
                                "box_rot": 0,  # 상자의 회전 각도 (0도로 설정)
                                "pallet_id": 1  # 팔레트 ID (1로 설정)
                            })

                            placed = True  # 상자가 배치되었음을 표시
                            break  # x축 반복 종료

        return out_placements  # 최종 배치 결과 반환
