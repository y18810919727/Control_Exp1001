def generator_replay_buffer(replay_type = "normal", size=10000):
    if replay_type is "normal":
        return ReplayBuffer(capacity=size)
    else:
        raise ValueError('No Replay named %s' % replay_type)
