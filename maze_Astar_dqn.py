# coding: utf-8
# 参考サイト：http://qiita.com/cvusk/items/e4f5862574c25649377a

import numpy as np
import pandas as pds
import random
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K
import sys


# 迷路に関する基本情報
class Maze(object):
    def __init__(self):
        # 壁の情報を含む迷路の配列
        #   0:進行可能 1:侵入不可
        #       全方位に壁がある -> 1111 -> 15
        #       全方位に壁がない -> 0000 -> 0
        #       右だけに壁がある -> 0100 -> 4
        #self.maze_list = [[ 9,  8, 14, 13, 13],
        #                    [ 7,  3, 12,  1,  4],
        #                    [ 9, 10,  2,  4,  5],
        #                    [ 5,  9, 12,  7,  5],
        #                    [ 3,  6,  3, 10,  6]]
        self.maze_list = [[9, 8, 8, 8, 8, 8, 8, 8, 8, 12],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                        [3, 2, 2, 2, 2, 2, 2, 2, 2, 6] ]


        # 各マスの報酬設定
        #   ゴールしたマスに100.0
        #   それ以外は0.0

        self.maze_eval = [[-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
                           [1.0, 0.0, 1.0, -100.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                           [-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0],
                           [1.0, -100.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
                           [-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -100.0, 1.0, -1.0],
                           [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                           [-1.0, -100.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0],
                           [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0],
                           [-1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0],
                           [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0]]

        self.start_point = (1, 1) # 開始地点の座標
        self.goal_point = (8, 8)  # ゴールの座標


# 環境の情報
# 迷路の情報，取った行動，報酬
class Field(object):
    def __init__(self, maze, mazeeval, start_point, goal_point):
        self.maze = maze # 迷路
        self.eval = mazeeval # 報酬
        self.start_point = start_point # 開始地点の座標
        self.goal_point = goal_point # ゴール地点の座標
        self.movable_vec = [[0, -1], [1, 0], [0, 1], [-1, 0]] # 現在位置から移動可能な距離

    # 現在の迷路の情報を表示
    def display(self, point=None):
        field_data = copy.deepcopy(self.maze)
        if not point is None:
                y, x = point
                field_data[y][x] = "@@" # エージェントの位置
        else:
                point = ""
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    # 移動可能な座標のリストを返す
    def get_actions(self, state):
        movables = []
        y, x = state

        for i in range(4):
            wall = bin(self.maze[y][x] >> i & 0b1) # 各方位の壁の情報
            y_ = state[0] + self.movable_vec[i][0]
            x_ = state[1] + self.movable_vec[i][1]

            if wall == "0b0": # 侵入可能であれば，移動候補に追加
                movables.append([y_, x_])

        if len(movables) != 0:
            return movables
        else:
            return None

    # 座標の報酬を返す
    def get_val(self, state):
        y, x = state
        if state == self.start_point: return 0, False
        else:
            v = float(self.eval[y][x])
            if state == self.goal_point:
                return v, True
            else:
                return v, False


# 状態：s
# 行動：a
# 報酬：r
# 次の状態：s'
# 次の行動：a'

# Deep Q-NeuralNetworkで迷路を解くためのクラス
    #sに対するaとrを実際に行動することで学習していく
class DQN_Solver:
    def __init__(self):
        self.memory = deque(maxlen=100000) # remember_memoryのキャパシティ
        self.gamma = 0.9 # 強化学習の割引率
        self.epsilon = 1.0 # ランダムな行動を取る確率
        self.e_decay = 0.9999 # 減衰率
        self.e_min = 0.01 # epsilonの最小値
        self.learning_rate = 0.0001 # 学習率
        self.model = self.build_model() # DQNモデル

    # DQNモデルの構築
    def build_model(self):
        model = Sequential()

        # 入力層
        #   2×2のデータ[(sのx座標, sのy座標), (a'のx座標，a'のy座標)]を受け取る
        #   ユニット数：128
        #   活性化関数：tanh
        model.add(Dense(128, input_shape=(2,2), activation='tanh'))
        # 2×2の入力を平滑化する
        #   2×2 -> 4
        model.add(Flatten())

        # 中間層
        #   ユニット数：128
        #   活性化関数：tanh
        model.add(Dense(128, activation='tanh'))
        #   ユニット数：128
        #   活性化関数：tanh
        model.add(Dense(128, activation='tanh'))

        # 出力層
        #   ユニット数：1
        #   活性化関数：linear
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
        return model

    # 学習のための各値をリストに格納
    #   state: 現在の座標
    #   action: 取ってみた行動
    #   reward: マスの報酬
    #   next_state: 行動後の座標
    #   next_movables: 行動後の可動域
    #   done: ゴールかどうか
    def remember_memory(self, state, action, reward, next_state, next_movables, done):
        self.memory.append((state, action, reward, next_state, next_movables, done))

    # 行動を決定する
        # εよりrandomが小さい -> ランダムに行動を選択
        # εよりrandomが大きい -> choose_best_action()でbestな行動選択
    def choose_action(self, state, movables):
        if self.epsilon >= random.random():
            return random.choice(movables)
        else:
            return self.choose_best_action(state, movables)

    # Neural Networkを用いて最も評価値の高い行動を選択する
    def choose_best_action(self, state, movables):
        best_actions = []
        max_act_value = -100

        for a in movables:
            np_action = np.array([[state, a]])
            # (現在の状態の座標，行動候補の座標)をNeural Networkに入力
            # 出力された値が候補となる行動の評価値
            act_value = self.model.predict(np_action)
            if act_value > max_act_value:
                best_actions = [a,]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)

        return random.choice(best_actions)

    # Neural Networkで現在の行動に対する期待値を近似
    def replay_experience(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            # memoryに格納されている各値を読み出す
            state, action, reward, next_state, next_movables, done = minibatch[i]
            input_action = [state, action]
            if done:
                target_f = reward
            else:
                next_rewards = []
                for i in next_movables:
                    # s'とa'から報酬の期待値をNeural Networkで計算
                    np_next_s_a = np.array([[next_state, i]])
                    next_rewards.append(self.model.predict(np_next_s_a))
                # 報酬の期待値の内一番高い値を選択
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + self.gamma * np_n_r_max
            # 行動と報酬の期待値を格納
            X.append(input_action)
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T

        # 行動と報酬の期待値を用いてmodelのfitting
        self.model.fit(np_X, np_Y, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            # εの減衰
            self.epsilon *= self.e_decay


def main(nepisode=10000, ntime=20000):
    # nepisode: モデルの学習回数
    # ntime: 1episodeにおける探索の回数

    # 迷路の生成
    mymaze = Maze()
    # 迷路の各マスの情報
    maze = mymaze.maze_list
    # 迷路の各環境の生成
    maze_field = Field(maze, mymaze.maze_eval, mymaze.start_point, mymaze.goal_point)
    # 迷路の表示
    maze_field.display()
    # DQNモデルの生成
    dql_solver = DQN_Solver()

    # モデルの学習
    for e in range(nepisode):
        print "episode: " + str(e)
        state = mymaze.start_point # 開始地点
        score = 0 # 報酬の合計値
        for time in range(ntime):
            movables = maze_field.get_actions(state) # 現在位置からの可動域の取得
            action = dql_solver.choose_action(state, movables) # 行動の決定
            reward, done = maze_field.get_val(action) # 報酬の取得
            score = score + reward # 報酬の合計値計算
            next_state = action # a'の決定
            next_movables = maze_field.get_actions(next_state) # a'の可動域の取得
            dql_solver.remember_memory(state, action, reward, next_state, next_movables, done) # 各値をリストに追加

            # ゴールの判定
            if done or time == (ntime - 1):
                # 500回ごとに情報を表示
                if e % 500 == 0:
                    print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
                            .format(e, nepisode, score, dql_solver.epsilon, time))
                break
            state = next_state # エージェントの移動
        dql_solver.replay_experience(32) # 期待値の近似

    # 学習した結果から迷路を解く
    state = mymaze.start_point
    score = 0 #報酬の合計値
    steps = 0 #ステップ数
    while True:
        # 現在位置からの可動域のリストを取得
        movables = maze_field.get_actions(state)
        # モデルから最良な行動を選択
        action = dql_solver.choose_best_action(state, movables)
        print("current state: {0} -> action: {1} ".format(state, action))

        # 報酬の取得とゴールの判定
        reward, done = maze_field.get_val(action)
        # エージェントの現在位置を表示
        maze_field.display(state)
        # 報酬の合計値を計算
        score = score + reward
        # エージェントの移動
        state = action
        print("current step: {0} \t score: {1}\n".format(steps, score))

        #ゴールした時
            # エージェントの位置とゴールの座標が一致
            # または，10,000ステップを行う
        if (state[0] == mymaze.goal_point[0] and state[1] == mymaze.goal_point[1]) or steps == 10000:
            maze_field.display(action)
            print("goal!")

            # モデルの学習に用いた各値の書き出し
            f = open('result.txt', 'w')
            for x in dql_solver.memory:
                f.write(str(x) + "\n")
            f.close()
            break

if __name__ == '__main__':
    main()
    #main(int(sys.argv[1]), int(sys.argv[2]))
