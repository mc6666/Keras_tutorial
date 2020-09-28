import random

# 環境類別
class Environment:
    # 初始化
    def __init__(self):
        # 最多走10步
        self.steps_left = 10

    def get_observation(self):
        # 狀態空間(State Space)
        return [0.0, 1.0, 2.0]

    def get_actions(self):
        # 行動空間(Action Space)
        return [0, 1]

    def is_done(self):
        # 回合(Episode)是否結束
        return self.steps_left == 0

    # 步驟
    def step(self, action):
        # 回合(Episode)結束
        if self.is_done():
            raise Exception("Game is over")
            
        # 減少1步
        self.steps_left -= 1
        
        # 隨機策略，任意行動，並給予獎勵(亂數值)
        return random.choice(self.get_observation()), random.random()


# 代理人類別
class Agent:
    # 初始化
    def __init__(self):
        pass
        
    def action(self, env):
        # 觀察或是取得狀態
        current_obs = env.get_observation()
        # 採取行動
        actions = env.get_actions()
        return random.choice(actions)


if __name__ == "__main__":
    # 實驗
    # 建立環境、代理人物件
    env = Environment()
    agent = Agent()

    # 累計報酬
    total_reward=0
    while not env.is_done():
        # 採取行動
        action = agent.action(env)
        
        # 進到下一步
        state, reward = env.step(action)
        
        # 報酬累計
        #print(reward)
        total_reward += reward
    
    # 顯示累計報酬
    print(f"累計報酬: {total_reward:.4f}")
