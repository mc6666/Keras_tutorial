import gym
import pandas as pd

# 載入遊戲
env = gym.make("CartPole-v0")

# 初始化
total_rewards = 0.0
total_steps = 0
obs = env.reset()

# 玩50回合
no = 50
all_steps=[]
all_rewards=[]
while True:
    # 隨機行動
    action = env.action_space.sample()
    # 進入下一步
    obs, reward, done, _ = env.step(action)
    # 渲染
    env.render()
    
    # 累計報酬
    total_rewards += reward
    # 累計步驟總數
    total_steps += 1
    if done:
        # 重置
        env.reset()
        
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
