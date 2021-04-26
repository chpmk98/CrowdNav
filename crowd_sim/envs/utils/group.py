from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Group(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.stdev = config.getfloat(section, 'stdev')

    # groups are only for instantiation so it doesn't do much else
