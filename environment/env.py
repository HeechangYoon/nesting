import copy
import random
import numpy as np
import math

from environment.data import read_data, generate_data
from environment.simulation import Management


class HiNEST(object):
    def __init__(self, look_ahead=2,  # 상태 구성 시 포함할 부재의 수
                 plate_l_min=3000,  # 생성될 강재의 최소 길이
                 plate_l_max=21000,  # 생성될 강재의 최대 길이
                 plate_b_min=1000,  # 생성될 강재의 최소 폭
                 plate_b_max=4500  # 생성될 강재의 최대 폭
                 ):
        self.look_ahead = look_ahead
        self.plate_l_min = plate_l_min
        self.plate_l_max = plate_l_max
        self.plate_b_min = plate_b_min
        self.plate_b_max = plate_b_max

        self.raw_part_list = read_data()

        plate, part_list = generate_data(self.raw_part_list,
                                         self.plate_l_min,
                                         self.plate_l_max,
                                         self.plate_b_min,
                                         self.plate_b_max
                                         )

        self.model = Management(plate, part_list)

        self.window = 5

        self.x_action_size = int(math.ceil(plate_l_max / 100) / self.window)
        self.y_action_size = int(math.ceil(plate_b_max / 100) / self.window)
        self.a_action_size = 24
        self.state_size = (plate.pixel_l_max, plate.pixel_b_max, look_ahead + 1)

    def step(self, action):
        done = False

        overlap = self.model.assign(action)

        if (len(self.model.part_list) == 0) | overlap:
            done = True

        reward, efficiency = self._calculate_reward(overlap, done)
        next_state = self._get_state()

        return next_state, reward, efficiency, done

    def reset(self):
        plate, part_list = generate_data(self.raw_part_list,
                                         self.plate_l_min,
                                         self.plate_l_max,
                                         self.plate_b_min,
                                         self.plate_b_max
                                         )

        self.model = Management(plate, part_list)

        return self._get_state()

    def get_possible_actions(self):
        possible_actions = []
        for i in range(len(self.model.part_list[0].PixelPart)):
            size = self.model.part_list[0].PixelPart[i].shape
            if (size[0] <= self.model.plate.pixel_l) & (size[1] <= self.model.plate.pixel_b):
                possible_actions.append(i)
        return possible_actions

    def _calculate_reward(self, overlap, done):
        reward = 0
        efficiency = 0
        if overlap:
            reward -= (len(self.model.part_list) + 1) / self.model.part_num
        if done:
            if len(self.model.part_list) + 1 != self.model.part_num:
                plate = self.model.plate
                plate_a = plate.PixelPlate[0:plate.pixel_l, 0:plate.pixel_b]

                non_zero_rows, non_zero_cols = np.nonzero(plate_a)
                start_row, start_col = non_zero_rows.min(), non_zero_cols.min()
                end_row, end_col = non_zero_rows.max() + 1, non_zero_cols.max() + 1
                assigned = plate_a[start_row:end_row, start_col:end_col]

                efficiency += np.sum(assigned) / ((end_row - start_row) * (end_col - start_col))
                reward += np.sum(assigned) / ((end_row - start_row) * (end_col - start_col))
        return reward, efficiency

    def _get_state(self):
        state = np.zeros(self.state_size)

        state[:, :, 0] = self.model.plate.PixelPlate

        for depth in range(self.look_ahead):
            temp = np.zeros(self.state_size[:2])

            if depth < len(self.model.part_list):
                i = 0
                while True:
                    part_shape = self.model.part_list[depth].PixelPart[i].shape
                    if (part_shape[0] <= self.state_size[0]) & (part_shape[1] <= self.state_size[1]):
                        break
                    i += 1

                start_row = (self.state_size[0] - part_shape[0]) // 2
                start_col = (self.state_size[1] - part_shape[1]) // 2

                temp[start_row:start_row + part_shape[0], start_col:start_col + part_shape[1]] = \
                self.model.part_list[depth].PixelPart[i]

            state[:, :, depth + 1] = temp

        return state


if __name__ == "__main__":
    plate_l_min = 3000
    plate_l_max = 21000
    plate_b_min = 1000
    plate_b_max = 4500

    nest = HiNEST(look_ahead=2,
                 plate_l_min=plate_l_min,
                 plate_l_max=plate_l_max,
                 plate_b_min=plate_b_min,
                 plate_b_max=plate_b_max
                 )

    s = nest.reset()

    r_cum = 0
    while True:
        possible_a = nest.get_possible_actions()
        part = nest.model.part_list[0]
        x = random.uniform(0, nest.model.plate.l)
        y = random.uniform(0, nest.model.plate.b)
        angle = random.choice(possible_a)
        a = (0, 0, angle)

        s_prime, r, efficiency, d = nest.step(a)
        print(d)
        r_cum += r
        print(r_cum)

        if d:
            break
