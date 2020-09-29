import gym
import pandas as pd
import math

# 載入遊戲
env = gym.make("CartPole-v0")

# 初始化
total_rewards = 0.0
total_steps = 0
observation = env.reset()

# 台車行進方向
left=0
right=1

max_angle = 8

# 代理人類別
class Agent:
    # 初始化
    def __init__(self):
        self.direction = left
        self.last_direction=right
        
    # 自訂策略
    def act(self, observation):
        # cart_position：台車位置(Cart Position)
        # cart_velocity：台車速度(Cart Velocity)
        # pole_angle：平衡桿角度(Pole Angle)
        # pole_velocity：平衡桿速度(Pole Velocity At Tip)
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        '''
        行動策略：
        1. 設定每次行動採一左一右，盡量不離中心點。
        2. 平衡桿角度偏右8度以上，就往右前進，直到角度偏右小於8度。
        3. 反之，偏左也是同樣處理。
        '''
        if pole_angle < math.radians(max_angle) and pole_angle > math.radians(-max_angle):
            self.direction = (self.last_direction + 1) % 2
        elif pole_angle >= math.radians(max_angle):
            self.direction = right
        else:
            self.direction = left

        self.last_direction = self.direction
        
        return self.direction    

# 玩50回合
no = 50
all_steps=[]
all_rewards=[]
agent = Agent()
while True:
    # 依策略行動
    action = agent.act(observation) #env.action_space.sample()
    # 進入下一步
    observation, reward, done, _ = env.step(action)
    # 渲染
    env.render()
    
    # 累計報酬
    total_rewards += reward
    # 累計步驟總數
    total_steps += 1
    if done:
        # 重置
        env.reset()
        agent = Agent()
        
        all_rewards.append(total_rewards)
        all_steps.append(total_steps)
        total_rewards = 0
        total_steps=0
        no-=1
        if no == 0:
            break

# 結束遊戲
env.close()

df = pd.DataFrame({'steps':all_steps, 'rewards':all_rewards})
print(df)
