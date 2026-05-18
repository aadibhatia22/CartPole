from models.agent_actions import AgentAction
from zl_al import ZlAl
class StateActionPair():
    def __init__(self, state, action:AgentAction, d_log_p_probability, zl_al:ZlAl):
        self.state = state
        self.actionTaken = action
        self.d_log_p_probability = d_log_p_probability
        self.ZlAl = zl_al
        self.reward = None

    def addReward(self, reward:float):
        self.reward = reward