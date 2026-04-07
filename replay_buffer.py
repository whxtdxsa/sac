import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=10**6):
        # self.buffer = deque(maxlen=capacity)
        self.state_buffer = torch.zeros(capacity, state_dim, dtype=torch.float32)
        self.action_buffer = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.reward_buffer = torch.zeros(capacity, 1, dtype=torch.float32)
        self.next_state_buffer = torch.zeros(capacity, state_dim, dtype=torch.float32)
        self.done_buffer = torch.zeros(capacity, 1, dtype=torch.bool)
        self.capacity = capacity
        self.index = 0
        self.len = 0

    def insert(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        self.state_buffer[self.index] = state
        self.action_buffer[self.index] = action
        self.reward_buffer[self.index] = reward
        self.next_state_buffer[self.index] = next_state
        self.done_buffer[self.index] = done
        self.index = (self.index + 1) % self.capacity
        self.len = min((self.len + 1), self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(low=0, high=self.len, size=(batch_size,))
        state = self.state_buffer[indices]
        action = self.action_buffer[indices]
        reward = self.reward_buffer[indices]
        next_state = self.next_state_buffer[indices]
        done = self.done_buffer[indices]

        return state, action, reward, next_state, done

    def __len__(self):
        return self.len
