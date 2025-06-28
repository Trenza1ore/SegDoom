from models.replay_memory import ReplayMemory

class DummyArray:
    def __setitem__(self, *args, **kwargs) -> None:
        return None
    def transpose(self, *args, **kwargs) -> None:
        return None
    def fill(self, *args, **kwargs) -> None:
        return None

class DummyMemory:
    def __init__(self) -> None:
        self._ptr = -1
        self.frames = self.features = self.rewards = self.actions = DummyArray()
    def add(self, *args, **kwargs) -> None:
        self._ptr += 1

class ReplayMemoryNoFrames:
    def __init__(self, *args, **kwargs) -> None:
        for fixed_value in ["res", "ch_num"]:
            if fixed_value in kwargs:
                del kwargs[fixed_value]
        mem = ReplayMemory(res=(1, 1), ch_num=1, **kwargs)
        mem.frames = DummyArray()
        self.dummy_frame = DummyArray()
        self._memory = mem
        self._ptr = self._memory._ptr
        self.features, self.rewards, self.actions = mem.features, mem.rewards, mem.actions
    def add(self, _, *args, **kwargs) -> None:
        self._memory.add(self.dummy_frame, *args, **kwargs)
        self._ptr = self._memory._ptr

def force_alloc_mem(mem: ReplayMemory | ReplayMemoryNoFrames | DummyMemory, val: object=0) -> None:
    mem.frames.fill(val)
    mem.features.fill(val)
    mem.rewards.fill(val)
    mem.actions.fill(val)
