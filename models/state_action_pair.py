from agent_action import AgentAction
class StateActionPair():
    def __init__(self, state, action:AgentAction, d_log_p_probability):
        self.state = state
        self.action = action
        self.d_log_p_probability = d_log_p_probability
