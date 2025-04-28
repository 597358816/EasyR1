from ..protocol import DataProto
import random
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Optional[DataProto] = None
        self.size = 0

    def add(self, new_data: DataProto):
        if self.buffer is None:
            self.buffer = new_data
            self.size = len(new_data)
        else:
            self.buffer = DataProto.concat([self.buffer, new_data])
            self.size += len(new_data)

        # 超出容量则裁剪
        if self.size > self.capacity:
            self._truncate_to_capacity()

    #def sample(self, batch_size: int) -> DataProto:
    #    if self.buffer is None or self.size == 0:
    #        raise ValueError("Replay buffer is empty")
    #    import random
    #    indices = random.sample(range(self.size), min(batch_size, self.size))
    #    return self.buffer.select(indices)

    def _truncate_to_capacity(self):
        """只保留最近的 capacity 条样本（假设时间顺序在 DataProto 中是保留的）"""
        assert self.buffer is not None
        indices = list(range(self.size - self.capacity, self.size))
        self.buffer = self.buffer.select_by_index(indices)
        self.size = self.capacity

    def __len__(self):
        return self.size
